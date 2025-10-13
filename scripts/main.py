"""
Main execution script for asset pricing simulation.

This script runs the complete simulation with unified panel generation
and analysis, correcting the mistake in Refactor v1 where they were separated.

Usage:
    python main.py --model gs --num-iters 10
    python main.py --config config.yaml
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from src.config import SimulationConfig
from src.panel_processor import PanelProcessor
from src.io import CSVResultsWriter
from src.utils import setup_logging, ProgressTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run asset pricing simulation with unified panel generation and analysis'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file (overrides other arguments)'
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['bgn', 'kp', 'gs'],
        default='gs',
        help='Theoretical model to use'
    )

    # Simulation parameters
    parser.add_argument(
        '--N',
        type=int,
        default=100,
        help='Number of firms'
    )
    parser.add_argument(
        '--T',
        type=int,
        default=400,
        help='Number of time periods (excluding burnin)'
    )
    parser.add_argument(
        '--num-iters',
        type=int,
        default=10,
        help='Number of Monte Carlo iterations'
    )
    parser.add_argument(
        '--start-iter',
        type=int,
        default=0,
        help='Starting iteration number'
    )

    # Computation parameters
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=10,
        help='Number of parallel jobs'
    )

    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )

    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=Path(args.output_dir) / 'simulation.log'
    )

    logger = logging.getLogger(__name__)

    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = SimulationConfig.from_yaml(Path(args.config))
    else:
        config = SimulationConfig(
            model=args.model,
            N=args.N,
            T=args.T,
            num_iters=args.num_iters,
            start_iter=args.start_iter,
            n_jobs=args.n_jobs,
            output_dir=Path(args.output_dir)
        )

    # Print configuration
    logger.info("=" * 60)
    logger.info("Asset Pricing Simulation - Refactor2")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model}")
    logger.info(f"Dimensions: N={config.N}, T={config.T}")
    logger.info(f"Iterations: {config.start_iter} to {config.start_iter + config.num_iters - 1}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Parallel jobs: {config.n_jobs}")
    logger.info("=" * 60)

    # Initialize results writer
    writer = CSVResultsWriter(config.output_dir)
    writer.initialize(N=config.N, model=config.model)

    # Create panel processor
    processor = PanelProcessor(config)

    # Setup progress tracking
    progress = ProgressTracker(total_iterations=config.num_iters)
    progress.start()

    # Run simulation with progress bar
    try:
        with tqdm(total=config.num_iters, desc="Iterations", unit="iter") as pbar:
            for i in range(config.num_iters):
                iter_num = config.start_iter + i

                # Process entire iteration: generation + analysis
                summary = processor.process_iteration(iter_num, writer)

                # Update progress bars
                pbar.set_postfix({
                    'months': summary.get('months_processed', 0),
                    'panel_size': summary.get('panel_shape', (0, 0))[0]
                })
                pbar.update(1)
                progress.update(i)

        # Finalize
        writer.finalize()
        progress.summary()

        logger.info("")
        logger.info("=" * 60)
        logger.info("Simulation complete!")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nSimulation interrupted by user")
        writer.finalize()
        return 1

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        writer.finalize()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
