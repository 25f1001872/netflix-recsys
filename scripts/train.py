"""
scripts/train.py
----------------
Unified training CLI for all models.

Usage:
    python scripts/train.py --model svd
    python scripts/train.py --model ncf --epochs 20
    python scripts/train.py --model als
    python scripts/train.py --model user_cf
    python scripts/train.py --model item_cf
    python scripts/train.py --model all          # train all models sequentially
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 🌟 FORCE SURPRISE TO STAY ON THE D: DRIVE
os.environ["SURPRISE_DATA_FOLDER"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".surprise_data")

import joblib
import yaml

from src.utils import resolve_path, config_to_absolute_paths, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

SUPPORTED_MODELS = ["svd", "als", "ncf", "user_cf", "item_cf", "hybrid", "all"]


def train_svd(train_df, val_df, cfg, output_dir):
    from src.models.svd_model import SVDRecommender
    model = SVDRecommender(**cfg["models"]["svd"])
    model.fit(train_df)
    model.save(os.path.join(output_dir, "svd.pkl"))
    return model


def train_als(train_df, val_df, cfg, output_dir):
    from src.models.als_model import ALSRecommender
    model = ALSRecommender(**cfg["models"]["als"])
    model.fit(train_df)
    model.save(os.path.join(output_dir, "als.pkl"))
    return model


def train_ncf(train_df, val_df, cfg, output_dir, encoder, override_epochs=None):
    from src.models.ncf_model import NCFRecommender
    ncf_cfg = dict(cfg["models"]["ncf"])
    if override_epochs:
        ncf_cfg["epochs"] = override_epochs

    model = NCFRecommender(
        n_users=encoder.n_users,
        n_items=encoder.n_movies,
        embedding_dim=ncf_cfg["embedding_dim"],
        layers=ncf_cfg["layers"],
        dropout=ncf_cfg["dropout"],
        learning_rate=ncf_cfg["learning_rate"],
        batch_size=ncf_cfg["batch_size"],
        epochs=ncf_cfg["epochs"],
    )
    model.fit(train_df, val_df=val_df, encoder=encoder)
    model.save(os.path.join(output_dir, "ncf.pkl"))
    return model


def train_user_cf(train_df, val_df, cfg, output_dir):
    from src.models.collaborative_filter import UserCFRecommender
    model = UserCFRecommender(**cfg["models"]["user_cf"])
    model.fit(train_df)
    model.save(os.path.join(output_dir, "user_cf.pkl"))
    return model


def train_item_cf(train_df, val_df, cfg, output_dir):
    from src.models.collaborative_filter import ItemCFRecommender
    model = ItemCFRecommender(**cfg["models"]["item_cf"])
    model.fit(train_df)
    model.save(os.path.join(output_dir, "item_cf.pkl"))
    return model


def train_hybrid(svd_model, ncf_model, train_df, val_df, cfg, output_dir):
    from src.models.hybrid_model import HybridRecommender
    weights = cfg["models"]["hybrid"]["weights"]
    model = HybridRecommender(
        models={"svd": svd_model, "ncf": ncf_model},
        blend_weights=weights,
        strategy="stack",
    )
    model.fit(train_df, val_df=val_df)
    model.save(os.path.join(output_dir, "hybrid.pkl"))
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Netflix Recommendation Models")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override NCF epochs")
    args = parser.parse_args()

    with open(resolve_path(args.config)) as f:
        cfg = yaml.safe_load(f)
    
    # Convert all relative paths to absolute
    cfg = config_to_absolute_paths(cfg)

    processed_dir = cfg["paths"]["processed_data"]
    output_dir = cfg["paths"]["models"]
    ensure_dir(output_dir)

    # Load data
    logger.info("Loading processed data ...")
    from src.data.loader import load_processed
    train_df, val_df, test_df = load_processed(processed_dir)
    encoder = joblib.load(os.path.join(processed_dir, "encoder.pkl"))

    logger.info(f"Training data: {len(train_df):,} ratings, "
                f"{encoder.n_users:,} users, {encoder.n_movies:,} movies")

    # Determine which models to train
    if args.model == "hybrid":
        # Special case: load SVD and NCF if they exist, train hybrid
        models_to_train = ["hybrid"]
    elif args.model == "all":
        models_to_train = ["svd", "als", "ncf", "user_cf", "item_cf", "hybrid"]
    else:
        models_to_train = [args.model]

    # Dispatch training
    trained_models = {}
    for model_name in models_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {model_name.upper()}")
        logger.info(f"{'='*50}")
        t0 = time.time()

        try:
            if model_name == "svd":
                trained_models["svd"] = train_svd(train_df, val_df, cfg, output_dir)
            elif model_name == "als":
                trained_models["als"] = train_als(train_df, val_df, cfg, output_dir)
            elif model_name == "ncf":
                trained_models["ncf"] = train_ncf(train_df, val_df, cfg, output_dir, encoder, args.epochs)
            elif model_name == "user_cf":
                trained_models["user_cf"] = train_user_cf(train_df, val_df, cfg, output_dir)
            elif model_name == "item_cf":
                trained_models["item_cf"] = train_item_cf(train_df, val_df, cfg, output_dir)
            elif model_name == "hybrid":
                # Load SVD and NCF (train them first if --model all)
                svd_path = os.path.join(output_dir, "svd.pkl")
                ncf_path = os.path.join(output_dir, "ncf.pkl")
                
                if not os.path.exists(svd_path) or not os.path.exists(ncf_path):
                    logger.error(
                        "Cannot train Hybrid: SVD and NCF models must exist.\n"
                        "Run: python scripts/train.py --model svd\n"
                        "     python scripts/train.py --model ncf\n"
                        "Or:  python scripts/train.py --model all"
                    )
                    continue
                
                svd = joblib.load(svd_path)
                ncf = joblib.load(ncf_path)
                trained_models["hybrid"] = train_hybrid(svd, ncf, train_df, val_df, cfg, output_dir)

            elapsed = time.time() - t0
            logger.info(f"✅  {model_name.upper()} trained in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"❌  Failed to train {model_name.upper()}: {e}", exc_info=True)
            if args.model == model_name:
                # User specifically requested this model; fail hard
                sys.exit(1)
            else:
                # Part of --all; skip this model and continue
                continue

    # Summary
    print(f"\n{'='*70}")
    print(f"✅  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Trained models: {', '.join(trained_models.keys()) or 'NONE'}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}")
    
    if not trained_models:
        logger.error("No models trained successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()