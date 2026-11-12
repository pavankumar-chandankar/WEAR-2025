import argparse

from data_loading import load_all_subjects

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge the challenge's sbj_*.csv files into one CSV")
    parser.add_argument('--data_dir', required=True,
                        help="Folder with the sbj_*.csv training files from the WEAR challenge")
    parser.add_argument('--save_path', default='processed_data.csv')
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}")
    df = load_all_subjects(data_dir=args.data_dir, save_processed=True, save_path=args.save_path)

    print(f"Loaded data shape: {df.shape}")
    print(f"Saved processed data to: {args.save_path}")
