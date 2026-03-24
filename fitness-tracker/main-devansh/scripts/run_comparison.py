# scripts/run_comparison.py
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.mediapipe_model import MediaPipeModel
from src.models.movenet_model import MoveNetModel
from src.analyzers.rep_counter import RepCounter
from src.analyzers.form_analyzer import FormAnalyzer


class ModelComparison:
    """Compare performance of different pose estimation models"""

    def __init__(self, video_dir: str, output_dir: str):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)

        # Create structured output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"comparison_run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for better organization
        self.results_dir = self.run_dir / "results"
        self.plots_dir = self.run_dir / "plots"
        self.videos_dir = self.run_dir / "annotated_videos"
        self.reports_dir = self.run_dir / "reports"

        for dir_path in [self.results_dir, self.plots_dir, self.videos_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)

        # Create model-specific subdirectories
        self.model_dirs = {}

        # Initialize models
        print("Initializing models...")
        self.models = {
            'mediapipe': MediaPipeModel(),
            'movenet_lightning': MoveNetModel('lightning'),
            'movenet_thunder': MoveNetModel('thunder'),
        }

        # Create directories for each model
        for model_name in self.models.keys():
            model_dir = self.videos_dir / model_name
            model_dir.mkdir(exist_ok=True)
            self.model_dirs[model_name] = model_dir
        self.results = []

    def scan_video_directory(self):
        """Dynamically scan video directory and detect all videos"""
        video_files = {}

        # Scan for video files
        for category_dir in self.video_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                video_files[category_name] = {}

                # Find all video files in this category
                for video_file in category_dir.glob("*.mp4"):
                    # Try to extract exercise type and expected reps from filename
                    filename = video_file.stem

                    # Parse filename (e.g., "squat_good_1" -> exercise="squat", reps=guess)
                    if 'squat' in filename.lower():
                        exercise = 'squat'
                        # Estimate reps based on video category
                        if 'good' in category_name:
                            reps = 8  # Default for good form videos
                        else:
                            reps = 5  # Default for bad form videos
                    elif 'shoulder_press' in filename.lower():
                        exercise = 'shoulder_press'
                        reps = 10
                    elif 'deadlift' in filename.lower():
                        exercise = 'deadlift'
                        reps = 6
                    elif 'pushup' in filename.lower():
                        exercise = 'pushup'
                        reps = 10
                    else:
                        exercise = 'squat'  # Default
                        reps = 5

                    video_files[category_name][video_file.name] = {
                        'exercise': exercise,
                        'reps': reps
                    }

        return video_files

    def process_video(self, video_path: Path, exercise_type: str, expected_reps: Optional[int] = None):
        """Process a single video with all models"""
        print(f"\nProcessing: {video_path.name}")

        results = {
            'video': video_path.name,
            'exercise': exercise_type,
            'expected_reps': expected_reps,
            'video_path': str(video_path),
            'processed_at': datetime.now().isoformat()
        }

        for model_name, model in self.models.items():
            print(f"  Running {model_name}...")

            # Reset model metrics
            model.reset_metrics()

            # Create model-specific output path
            output_path = self.model_dirs[model_name] / f"{video_path.stem}_{model_name}.mp4"

            # Process video
            pose_results = model.process_video(str(video_path), str(output_path))

            # Analyze with rep counter and form analyzer
            rep_counter = RepCounter(exercise_type)
            form_analyzer = FormAnalyzer(exercise_type)

            for i, frame_result in enumerate(pose_results):
                if frame_result['keypoints'].shape[0] > 0:
                    keypoints = frame_result['keypoints'][0]  # First person

                    # Count reps
                    rep_result = rep_counter.process_frame(keypoints, i)

                    # Analyze form
                    form_faults = form_analyzer.analyze_form(
                        keypoints,
                        rep_result['current_state']
                    )

            # Get metrics
            performance_metrics = model.get_performance_metrics()
            rep_metrics = rep_counter.get_rep_quality_metrics()
            form_summary = form_analyzer.get_form_summary()

            # Store results
            results[model_name] = {
                'performance': performance_metrics,
                'reps': rep_metrics,
                'form': form_summary,
                'detected_reps': rep_counter.rep_count,
                'accuracy': (rep_counter.rep_count / expected_reps * 100) if expected_reps else None,
                'annotated_video_path': str(output_path)
            }

            # Save individual model results
            model_result_file = self.results_dir / f"{video_path.stem}_{model_name}_results.json"
            with open(model_result_file, 'w') as f:
                json.dump(results[model_name], f, indent=2)

            # Save rep counter plot
            plot_path = self.plots_dir / f"{video_path.stem}_{model_name}_reps.png"
            rep_counter.plot_angle_trace(str(plot_path))

        self.results.append(results)

        # Save individual video comparison
        video_comparison_file = self.results_dir / f"{video_path.stem}_comparison.json"
        with open(video_comparison_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_comparison(self):
        """Run comparison on all videos in the directory"""
        # Dynamically scan video directory
        test_videos = self.scan_video_directory()

        print(f"Found video categories: {list(test_videos.keys())}")

        # Process each category
        for category, videos in test_videos.items():
            print(f"\n=== Processing {category} videos ===")
            category_dir = self.video_dir / category

            if not category_dir.exists():
                print(f"Warning: {category_dir} not found")
                continue

            for video_file, info in videos.items():
                video_path = category_dir / video_file
                if video_path.exists():
                    self.process_video(
                        video_path,
                        info['exercise'],
                        info['reps']
                    )
                else:
                    print(f"Warning: {video_path} not found")

    def generate_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results to report")
            return

        print(f"\nGenerating comprehensive report...")

        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            for model_name in self.models.keys():
                if model_name in result:
                    model_data = result[model_name]
                    summary_data.append({
                        'video': result['video'],
                        'exercise': result['exercise'],
                        'model': model_name,
                        'fps': model_data['performance'].get('avg_fps', 0),
                        'processing_time': model_data['performance'].get('avg_processing_time', 0),
                        'detected_reps': model_data['detected_reps'],
                        'expected_reps': result['expected_reps'],
                        'rep_accuracy': model_data['accuracy'],
                        'form_score': model_data['form']['form_score'],
                        'total_faults': model_data['form']['total_faults']
                    })

        df = pd.DataFrame(summary_data)

        # Save all results in structured format
        self._save_structured_results(df)

        # Generate visualizations
        self._create_performance_plots(df)
        self._create_accuracy_plots(df)
        self._create_form_analysis_plots(df)

        # Generate text report
        self._generate_text_report(df)

        # Create run summary
        self._create_run_summary()

        print(f"Complete report saved to: {self.run_dir}")

    def _save_structured_results(self, df: pd.DataFrame):
        """Save results in multiple structured formats"""

        # 1. Complete detailed results
        detailed_file = self.results_dir / 'complete_results.json'
        with open(detailed_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # 2. Summary CSV
        summary_file = self.results_dir / 'summary_results.csv'
        df.to_csv(summary_file, index=False)

        # 3. Per-model summaries
        for model_name in self.models.keys():
            model_df = df[df['model'] == model_name]
            if not model_df.empty:
                model_file = self.results_dir / f'{model_name}_summary.csv'
                model_df.to_csv(model_file, index=False)

        # 4. Per-exercise summaries
        for exercise in df['exercise'].unique():
            exercise_df = df[df['exercise'] == exercise]
            if not exercise_df.empty:
                exercise_file = self.results_dir / f'{exercise}_summary.csv'
                exercise_df.to_csv(exercise_file, index=False)

        # 5. Performance metrics only
        performance_df = df[['video', 'model', 'fps', 'processing_time']].copy()
        performance_file = self.results_dir / 'performance_metrics.csv'
        performance_df.to_csv(performance_file, index=False)

        print(f"Results saved in multiple formats to: {self.results_dir}")

    def _create_performance_plots(self, df: pd.DataFrame):
        """Create performance comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # FPS comparison
        sns.boxplot(data=df, x='model', y='fps', ax=axes[0])
        axes[0].set_title('FPS Comparison')
        plt.setp(axes[0].get_xticklabels(), rotation=45)

        # Processing time comparison
        sns.boxplot(data=df, x='model', y='processing_time', ax=axes[1])
        axes[1].set_title('Processing Time per Frame (seconds)')
        plt.setp(axes[1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_accuracy_plots(self, df: pd.DataFrame):
        """Create accuracy comparison plots"""
        df_with_reps = df[df['expected_reps'].notna()]

        if df_with_reps.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        model_accuracy = df_with_reps.groupby('model')['rep_accuracy'].mean()
        bars = model_accuracy.plot(kind='bar', ax=ax)
        ax.set_title('Average Rep Counting Accuracy by Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 110)

        for i, v in enumerate(model_accuracy):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rep_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_form_analysis_plots(self, df: pd.DataFrame):
        """Create form analysis comparison plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Form score comparison
        sns.boxplot(data=df, x='model', y='form_score', ax=axes[0])
        axes[0].set_title('Form Score Comparison')
        axes[0].set_ylim(0, 105)
        plt.setp(axes[0].get_xticklabels(), rotation=45)

        # Fault detection comparison
        fault_avg = df.groupby('model')['total_faults'].mean()
        fault_avg.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Average Faults Detected per Video')
        axes[1].set_ylabel('Number of Faults')
        plt.setp(axes[1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'form_analysis_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_text_report(self, df: pd.DataFrame):
        """Generate detailed text report"""
        report_file = self.reports_dir / 'comparison_report.md'

        report = []
        report.append("# Pose Estimation Model Comparison Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Total videos analyzed:** {len(self.results)}\n")
        report.append(f"**Models compared:** {', '.join(self.models.keys())}\n")

        # Performance summary
        report.append("\n## Performance Summary\n")
        for model in self.models.keys():
            model_df = df[df['model'] == model]
            if not model_df.empty:
                report.append(f"\n### {model.replace('_', ' ').title()}")
                report.append(f"- **Average FPS:** {model_df['fps'].mean():.1f} (\u00B1{model_df['fps'].std():.1f})")
                report.append(f"- **Average processing time:** {model_df['processing_time'].mean():.3f}s")
                report.append(f"- **Rep counting accuracy:** {model_df['rep_accuracy'].mean():.1f}%")
                report.append(f"- **Average form score:** {model_df['form_score'].mean():.1f}")
        # Key findings
        report.append("\n##Key Findings\n")

        best_fps_model = df.groupby('model')['fps'].mean().idxmax()
        report.append(
            f"- **Fastest model:** {best_fps_model} ({df[df['model'] == best_fps_model]['fps'].mean():.1f} FPS)")

        df_with_accuracy = df[df['rep_accuracy'].notna()]
        if not df_with_accuracy.empty:
            best_accuracy_model = df_with_accuracy.groupby('model')['rep_accuracy'].mean().idxmax()
            report.append(
                f"- **Most accurate rep counting:** {best_accuracy_model} ({df_with_accuracy[df_with_accuracy['model'] == best_accuracy_model]['rep_accuracy'].mean():.1f}%)")

        # Write report
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"Report saved to: {report_file}")

    def _create_run_summary(self):
        """Create a summary of this comparison run"""
        summary = {
            'run_timestamp': self.timestamp,
            'run_directory': str(self.run_dir),
            'total_videos_processed': len(self.results),
            'models_tested': list(self.models.keys()),
            'video_categories': list(set(r['exercise'] for r in self.results)),
            'output_structure': {
                'results': str(self.results_dir),
                'plots': str(self.plots_dir),
                'annotated_videos': str(self.videos_dir),
                'reports': str(self.reports_dir)
            },
            'files_generated': {
                'detailed_results': 'results/complete_results.json',
                'summary_csv': 'results/summary_results.csv',
                'performance_plot': 'plots/performance_comparison.png',
                'accuracy_plot': 'plots/rep_accuracy_comparison.png',
                'form_plot': 'plots/form_analysis_comparison.png',
                'report': 'reports/comparison_report.md'
            }
        }

        summary_file = self.run_dir / 'run_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also create a README for this run
        readme_content = f"""# Model Comparison Run - {self.timestamp}

## Overview
This directory contains the complete results from a pose estimation model comparison run.

## Structure
- `results/` - All numerical results and data files
- `plots/` - Visualization and comparison charts  
- `annotated_videos/` - Processed videos with pose annotations (organized by model)
- `reports/` - Human-readable reports and summaries

## Models Tested
{chr(10).join(f"- {model}" for model in self.models.keys())}

## Videos Processed
Total: {len(self.results)} videos

## Quick Access
- **Main Report:** [reports/comparison_report.md](reports/comparison_report.md)
- **Summary Data:** [results/summary_results.csv](results/summary_results.csv)
- **Performance Chart:** [plots/performance_comparison.png](plots/performance_comparison.png)
"""

        readme_file = self.run_dir / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)


def main():
    parser = argparse.ArgumentParser(description='Compare pose estimation models')
    parser.add_argument('--video-dir', type=str, default='data/videos',
                        help='Directory containing test videos')
    parser.add_argument('--output-dir', type=str, default='data/results',
                        help='Directory to save results')
    parser.add_argument('--single-video', type=str,
                        help='Process a single video file')
    parser.add_argument('--exercise', type=str, default='squat',
                        choices=['squat', 'shoulder_press', 'deadlift', 'pushup'],
                        help='Exercise type for single video')
    parser.add_argument('--reps', type=int,
                        help='Expected number of reps for single video')

    args = parser.parse_args()

    # Create comparison instance
    comparison = ModelComparison(args.video_dir, args.output_dir)

    if args.single_video:
        # Process single video
        video_path = Path(args.single_video)
        if video_path.exists():
            comparison.process_video(video_path, args.exercise, args.reps)
            comparison.generate_report()
        else:
            print(f"Error: Video file {video_path} not found")
    else:
        # Run full comparison
        comparison.run_comparison()
        comparison.generate_report()


if __name__ == "__main__":
    main()