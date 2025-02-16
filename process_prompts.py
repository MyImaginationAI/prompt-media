#!/usr/bin/env python3

import os
import subprocess
import argparse
from typing import List, Dict
import yaml
import time

class PromptProcessor:
    def __init__(self, categories_path: str, workflow: str):
        self.categories_path = categories_path
        self.workflow = workflow

    def get_yaml_files(self) -> Dict[str, List[str]]:
        """Get all YAML files organized by category."""
        categories = {}
        for category in os.listdir(self.categories_path):
            category_path = os.path.join(self.categories_path, category)
            if os.path.isdir(category_path):
                yaml_files = []
                for file in os.listdir(category_path):
                    if file.endswith('.yaml'):
                        yaml_files.append(os.path.join(category_path, file))
                if yaml_files:
                    categories[category] = sorted(yaml_files)
        return categories

    def process_file(self, file_path: str) -> None:
        """Process a single prompt media file."""
        relative_path = os.path.relpath(file_path)
        cmd = ['make', f'flux/{self.workflow}', f'PROMPT_MEDIA_FILE={relative_path}']
        print(f"\n🔄 Processing {os.path.basename(file_path)} with workflow: {self.workflow}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Successfully processed {os.path.basename(file_path)}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")

    def process_category(self, category: str, files: List[str]) -> None:
        """Process all files in a category."""
        print(f"\n📁 Processing category: {category}")
        print("=" * 50)
        for file in files:
            self.process_file(file)
            time.sleep(1)  # Small delay between files

    def process_all(self, selected_categories: List[str] = None) -> None:
        """Process all categories or selected ones."""
        categories = self.get_yaml_files()
        
        if selected_categories:
            categories = {k: v for k, v in categories.items() if k in selected_categories}

        print(f"🚀 Starting prompt processing with workflow: {self.workflow}")
        print(f"Found {len(categories)} categories")
        
        for category, files in categories.items():
            self.process_category(category, files)

def main():
    parser = argparse.ArgumentParser(description='Process prompt media files.')
    parser.add_argument('--categories', nargs='*', help='Specific categories to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without executing')
    workflow_group = parser.add_mutually_exclusive_group(required=True)
    workflow_group.add_argument('--dev', action='store_true', help='Use development workflow')
    workflow_group.add_argument('--schnell', action='store_true', help='Use schnell workflow')
    args = parser.parse_args()

    workflow = 'dev' if args.dev else 'schnell'
    categories_path = os.path.join('collections', 'prompts', 'categories')
    processor = PromptProcessor(categories_path, workflow)
    
    if args.dry_run:
        categories = processor.get_yaml_files()
        print("\n📋 Files that would be processed:")
        for category, files in categories.items():
            if not args.categories or category in args.categories:
                print(f"\n{category}:")
                for f in files:
                    print(f"  - {os.path.basename(f)} ({workflow})")
    else:
        processor.process_all(args.categories)

if __name__ == "__main__":
    main()
