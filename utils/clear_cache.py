#!/usr/bin/env python3
"""
Script to clear Hugging Face model cache to fix size mismatch errors.
Run this before executing your main script if you encounter model loading issues.
"""

import os
import shutil
from pathlib import Path


def get_cache_dir():
    """Get Hugging Face cache directory based on OS."""
    home = Path.home()
    
    # Check common cache locations
    possible_paths = [
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface" / "transformers",
        home / "Library" / "Caches" / "huggingface" / "hub",  # macOS alternative
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def clear_qwen_cache():
    """Clear only Qwen model cache."""
    cache_dir = get_cache_dir()
    
    if cache_dir is None:
        print("❌ Could not find Hugging Face cache directory")
        return False
    
    print(f"📂 Cache directory: {cache_dir}")
    print()
    
    # Find all Qwen model directories
    qwen_dirs = list(cache_dir.glob("models--Qwen--*"))
    
    if not qwen_dirs:
        print("✅ No Qwen models found in cache (cache is clean)")
        return True
    
    print(f"Found {len(qwen_dirs)} Qwen model(s) in cache:")
    for qwen_dir in qwen_dirs:
        size_mb = sum(f.stat().st_size for f in qwen_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        print(f"  - {qwen_dir.name} ({size_mb:.1f} MB)")
    
    print()
    response = input("⚠️  Delete these cached models? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print()
        for qwen_dir in qwen_dirs:
            try:
                print(f"🗑️  Deleting: {qwen_dir.name}")
                shutil.rmtree(qwen_dir)
                print(f"✅ Deleted successfully")
            except Exception as e:
                print(f"❌ Error deleting {qwen_dir.name}: {e}")
        
        print()
        print("🎉 Cache cleared! Your model will re-download on next run.")
        return True
    else:
        print("❌ Cache clearing cancelled")
        return False


def clear_all_cache():
    """Clear entire Hugging Face cache (nuclear option)."""
    cache_dir = get_cache_dir()
    
    if cache_dir is None:
        print("❌ Could not find Hugging Face cache directory")
        return False
    
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / (1024 * 1024 * 1024)
    
    print(f"📂 Cache directory: {cache_dir}")
    print(f"📊 Total cache size: {total_size:.2f} GB")
    print()
    print("⚠️  WARNING: This will delete ALL cached Hugging Face models!")
    response = input("Are you sure? (type 'DELETE ALL' to confirm): ").strip()
    
    if response == "DELETE ALL":
        try:
            print()
            print("🗑️  Deleting entire cache...")
            shutil.rmtree(cache_dir)
            print("✅ All cache cleared!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    else:
        print("❌ Operation cancelled")
        return False


def main():
    print("=" * 60)
    print("🧹 HUGGING FACE CACHE CLEANER")
    print("=" * 60)
    print()
    print("This script helps fix 'size mismatch' errors by clearing")
    print("corrupted or mismatched model caches.")
    print()
    print("Choose an option:")
    print("1. Clear only Qwen models (recommended)")
    print("2. Clear entire Hugging Face cache (nuclear option)")
    print("3. Exit")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    print()
    
    if choice == "1":
        clear_qwen_cache()
    elif choice == "2":
        clear_all_cache()
    elif choice == "3":
        print("👋 Exiting...")
    else:
        print("❌ Invalid choice")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()