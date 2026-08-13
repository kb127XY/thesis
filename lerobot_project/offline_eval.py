import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch


# Compatible imports for different LeRobot versions.
try:
    from lerobot.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDatasetMetadata = None


try:
    from lerobot.policies import make_pre_post_processors
except ImportError:
    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError:
        make_pre_post_processors = None


try:
    from lerobot.policies.act.modeling_act import ACTPolicy
except ImportError:
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise ImportError(
            "ACTPolicy could not be imported. Check the installed "
            "LeRobot package structure."
        ) from exc


class IdentityProcessor:
    """Fallback processor used when no LeRobot processor is available."""

    def __call__(self, value):
        return value


def safe_make_out_dir(out_dir: str) -> Path:
    """Create the output directory, with a desktop fallback."""
    out_path = Path(out_dir)

    try:
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path
    except PermissionError:
        fallback = Path.home() / "Desktop" / out_path.name
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[Warning] Cannot write to {out_path}.")
        print(f"[Warning] Using {fallback} instead.")
        return fallback


def scalar_value(value, default=-1):
    """Convert a tensor, NumPy scalar, or Python scalar."""
    if value is None:
        return default

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return default

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return default

    try:
        return value.item()
    except AttributeError:
        return value


def get_dataset_stats(repo_id, dataset):
    """Load normalisation statistics across supported LeRobot APIs."""
    if LeRobotDatasetMetadata is not None:
        try:
            metadata = LeRobotDatasetMetadata(repo_id)
            if hasattr(metadata, "stats"):
                print("[Info] Loaded stats from dataset metadata.")
                return metadata.stats
        except Exception as exc:
            print(f"[Warning] Metadata loading failed: {exc}")

    for attr_name in ("stats", "meta", "metadata"):
        if not hasattr(dataset, attr_name):
            continue

        attribute = getattr(dataset, attr_name)
        if attr_name == "stats":
            print("[Info] Loaded stats from dataset.stats.")
            return attribute

        if hasattr(attribute, "stats"):
            print(f"[Info] Loaded stats from dataset.{attr_name}.stats.")
            return attribute.stats

        if isinstance(attribute, dict) and "stats" in attribute:
            print(f"[Info] Loaded stats from dataset.{attr_name}.")
            return attribute["stats"]

    print("[Warning] Dataset stats were not found.")
    return None


def build_pre_post_processors(policy, dataset_stats):
    """Construct compatible LeRobot pre- and post-processors."""
    if make_pre_post_processors is None or not hasattr(policy, "config"):
        print("[Warning] Using identity pre/post processors.")
        return IdentityProcessor(), IdentityProcessor()

    attempts = [
        lambda: make_pre_post_processors(
            policy.config, dataset_stats=dataset_stats
        ),
        lambda: make_pre_post_processors(
            policy_cfg=policy.config, dataset_stats=dataset_stats
        ),
        lambda: make_pre_post_processors(
            cfg=policy.config, dataset_stats=dataset_stats
        ),
        lambda: make_pre_post_processors(
            policy.config, stats=dataset_stats
        ),
        lambda: make_pre_post_processors(
            policy_cfg=policy.config, stats=dataset_stats
        ),
        lambda: make_pre_post_processors(
            cfg=policy.config, stats=dataset_stats
        ),
    ]

    for attempt in attempts:
        try:
            processors = attempt()
            if isinstance(processors, tuple) and len(processors) == 2:
                return processors

            if isinstance(processors, dict):
                preprocess = processors.get(
                    "preprocess", IdentityProcessor()
                )
                postprocess = processors.get(
                    "postprocess", IdentityProcessor()
                )
                return preprocess, postprocess
        except TypeError:
            continue
        except Exception as exc:
            print(f"[Warning] Processor construction failed: {exc}")

    print("[Warning] Using identity pre/post processors.")
    return IdentityProcessor(), IdentityProcessor()


def get_policy_input_keys(policy, sample):
    """Infer the observation keys expected by the policy."""
    if hasattr(policy, "config") and hasattr(
        policy.config, "input_features"
    ):
        try:
            keys = [
                key
                for key in policy.config.input_features
                if key in sample
            ]
            if keys:
                return keys
        except Exception:
            pass

    return [
        key for key in sample
        if key.startswith("observation.")
    ]


def to_device_batch(sample, device, input_keys):
    """Create a batch of one sample on the selected device."""
    batch = {}

    for key in input_keys:
        if key not in sample:
            continue

        value = sample[key]
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value

    return batch


def extract_action_tensor(action):
    """Convert a policy output to a one-dimensional NumPy array."""
    if isinstance(action, dict):
        if "action" not in action:
            raise KeyError(
                f"No action key in policy output: {action.keys()}"
            )
        action = action["action"]

    if torch.is_tensor(action):
        action = action.detach().cpu().float().numpy()
    else:
        action = np.asarray(action, dtype=np.float32)

    if action.ndim == 3:
        action = action[0, 0]
    elif action.ndim == 2:
        action = action[0]
    elif action.ndim != 1:
        action = action.reshape(-1)

    return action.astype(np.float32)


def extract_gt_action(sample):
    """Extract the recorded demonstration action from one sample."""
    if "action" not in sample:
        raise KeyError("The dataset sample has no action field.")

    action = sample["action"]
    if torch.is_tensor(action):
        action = action.detach().cpu().float().numpy()
    else:
        action = np.asarray(action, dtype=np.float32)

    if action.ndim > 1:
        action = action.reshape(-1, action.shape[-1])[0]

    return action.astype(np.float32)


def episode_slices(episode_ids):
    """Return contiguous slices without crossing episode boundaries."""
    if len(episode_ids) == 0:
        return []

    boundaries = np.flatnonzero(
        episode_ids[1:] != episode_ids[:-1]
    ) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_ids)]))
    return [slice(start, end) for start, end in zip(starts, ends)]


def temporal_variation_scores(actions, episode_ids):
    """Compute first- and second-order differences per episode."""
    first_order = []
    second_order = []

    for episode_slice in episode_slices(episode_ids):
        sequence = actions[episode_slice]

        if len(sequence) >= 2:
            delta = np.diff(sequence, axis=0)
            first_order.append(np.mean(np.abs(delta), axis=1))

        if len(sequence) >= 3:
            second_delta = (
                sequence[2:]
                - 2.0 * sequence[1:-1]
                + sequence[:-2]
            )
            second_order.append(
                np.mean(np.abs(second_delta), axis=1)
            )

    first = (
        np.concatenate(first_order)
        if first_order
        else np.asarray([math.nan])
    )
    second = (
        np.concatenate(second_order)
        if second_order
        else np.asarray([math.nan])
    )
    return first, second


def compute_delay_frames(
    pred, gt, episode_ids, max_lag=15
):
    """Find the lag that minimises the episode-safe global L1 error."""
    best_lag = 0
    best_error = float("inf")

    for lag in range(-max_lag, max_lag + 1):
        absolute_error_sum = 0.0
        compared_values = 0

        for episode_slice in episode_slices(episode_ids):
            pred_episode = pred[episode_slice]
            gt_episode = gt[episode_slice]

            if lag < 0:
                pred_aligned = pred_episode[:lag]
                gt_aligned = gt_episode[-lag:]
            elif lag > 0:
                pred_aligned = pred_episode[lag:]
                gt_aligned = gt_episode[:-lag]
            else:
                pred_aligned = pred_episode
                gt_aligned = gt_episode

            if len(pred_aligned) < 2:
                continue

            absolute_error_sum += float(
                np.abs(pred_aligned - gt_aligned).sum()
            )
            compared_values += pred_aligned.size

        if compared_values == 0:
            continue

        error = absolute_error_sum / compared_values
        if error < best_error:
            best_error = error
            best_lag = lag

    return best_lag, best_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument(
        "--out-dir", default="./offline_eval_output"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--gripper-index", type=int, default=-1)
    parser.add_argument(
        "--reset-every-frame", action="store_true"
    )
    args = parser.parse_args()

    out_dir = safe_make_out_dir(args.out_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset = LeRobotDataset(args.repo_id)
    sample0 = dataset[0]
    dataset_stats = get_dataset_stats(args.repo_id, dataset)

    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()

    preprocess, postprocess = build_pre_post_processors(
        policy, dataset_stats
    )
    input_keys = get_policy_input_keys(policy, sample0)

    records = []
    pred_actions = []
    gt_actions = []
    episode_ids = []

    sample_count = len(dataset)
    if args.max_samples > 0:
        sample_count = min(sample_count, args.max_samples)

    last_episode = None

    with torch.inference_mode():
        for index in range(sample_count):
            sample = dataset[index]
            gt_action = extract_gt_action(sample)

            episode_id = int(
                scalar_value(
                    sample.get("episode_index", -1),
                    default=-1,
                )
            )
            timestamp = float(
                scalar_value(
                    sample.get("timestamp", index),
                    default=index,
                )
            )

            if episode_id != last_episode:
                if hasattr(policy, "reset"):
                    policy.reset()
                last_episode = episode_id

            # Enable this option for independent, frame-wise
            # teacher-forced predictions.
            if args.reset_every_frame and hasattr(policy, "reset"):
                policy.reset()

            batch = to_device_batch(
                sample, device, input_keys
            )

            try:
                batch = preprocess(batch)
            except Exception as exc:
                print(
                    f"[Warning] Preprocess failed at "
                    f"frame {index}: {exc}"
                )

            pred_action = policy.select_action(batch)

            try:
                pred_action = postprocess(pred_action)
            except Exception as exc:
                print(
                    f"[Warning] Postprocess failed at "
                    f"frame {index}: {exc}"
                )

            pred_action = extract_action_tensor(pred_action)

            dimension = min(len(pred_action), len(gt_action))
            pred_action = pred_action[:dimension]
            gt_action = gt_action[:dimension]
            absolute_error = np.abs(pred_action - gt_action)

            action_l1 = float(np.mean(absolute_error))
            gripper_index = args.gripper_index
            if gripper_index < 0:
                gripper_index = dimension + gripper_index

            if 0 <= gripper_index < dimension:
                gripper_l1 = float(
                    absolute_error[gripper_index]
                )
                arm_l1 = (
                    float(
                        np.mean(
                            np.delete(
                                absolute_error,
                                gripper_index,
                            )
                        )
                    )
                    if dimension > 1
                    else math.nan
                )
            else:
                gripper_l1 = math.nan
                arm_l1 = action_l1

            records.append(
                {
                    "index": index,
                    "episode_index": episode_id,
                    "timestamp": timestamp,
                    "action_l1": action_l1,
                    "arm_l1": arm_l1,
                    "gripper_l1": gripper_l1,
                }
            )
            pred_actions.append(pred_action)
            gt_actions.append(gt_action)
            episode_ids.append(episode_id)

            if index % 500 == 0:
                print(
                    f"[Info] Processed {index}/{sample_count}; "
                    f"action_l1={action_l1:.6f}"
                )

    pred_actions = np.stack(pred_actions)
    gt_actions = np.stack(gt_actions)
    episode_ids = np.asarray(episode_ids)

    pred_smoothness, pred_jerk = (
        temporal_variation_scores(
            pred_actions, episode_ids
        )
    )
    gt_smoothness, gt_jerk = (
        temporal_variation_scores(
            gt_actions, episode_ids
        )
    )

    best_lag, best_lag_error = compute_delay_frames(
        pred_actions,
        gt_actions,
        episode_ids,
        max_lag=15,
    )

    csv_path = out_dir / "offline_replay_metrics.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "index",
                "episode_index",
                "timestamp",
                "action_l1",
                "arm_l1",
                "gripper_l1",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    action_l1_values = np.asarray(
        [record["action_l1"] for record in records],
        dtype=np.float32,
    )
    arm_l1_values = np.asarray(
        [record["arm_l1"] for record in records],
        dtype=np.float32,
    )
    gripper_l1_values = np.asarray(
        [record["gripper_l1"] for record in records],
        dtype=np.float32,
    )

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as summary_file:
        summary_file.write(f"repo_id: {args.repo_id}\n")
        summary_file.write(
            f"policy_path: {args.policy_path}\n"
        )
        summary_file.write(
            f"num_samples: {len(records)}\n"
        )
        summary_file.write(f"device: {device}\n")
        summary_file.write(
            f"reset_every_frame: "
            f"{args.reset_every_frame}\n\n"
        )

        summary_file.write(
            f"mean_action_l1: "
            f"{np.nanmean(action_l1_values):.6f}\n"
        )
        summary_file.write(
            f"std_action_l1: "
            f"{np.nanstd(action_l1_values):.6f}\n"
        )
        summary_file.write(
            f"mean_arm_l1: "
            f"{np.nanmean(arm_l1_values):.6f}\n"
        )
        summary_file.write(
            f"std_arm_l1: "
            f"{np.nanstd(arm_l1_values):.6f}\n"
        )
        summary_file.write(
            f"mean_gripper_l1: "
            f"{np.nanmean(gripper_l1_values):.6f}\n"
        )
        summary_file.write(
            f"std_gripper_l1: "
            f"{np.nanstd(gripper_l1_values):.6f}\n\n"
        )

        summary_file.write(
            f"mean_pred_smoothness: "
            f"{np.nanmean(pred_smoothness):.6f}\n"
        )
        summary_file.write(
            f"mean_gt_smoothness: "
            f"{np.nanmean(gt_smoothness):.6f}\n"
        )
        summary_file.write(
            f"mean_pred_jerk: "
            f"{np.nanmean(pred_jerk):.6f}\n"
        )
        summary_file.write(
            f"mean_gt_jerk: "
            f"{np.nanmean(gt_jerk):.6f}\n\n"
        )

        summary_file.write(
            f"best_delay_lag_frames: {best_lag}\n"
        )
        summary_file.write(
            f"best_delay_lag_error: "
            f"{best_lag_error:.6f}\n"
        )

    print("[Done] Offline replay evaluation finished.")
    print(f"[Done] Metrics CSV: {csv_path}")
    print(f"[Done] Summary: {summary_path}")


if __name__ == "__main__":
    main()