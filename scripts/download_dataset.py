"""
Download COCO128 dataset using Ultralytics built-in downloader.
COCO128 is a mini version of COCO with 128 images and 80 classes (~7MB).
"""
import sys


def download_coco128():
    print("Downloading COCO128 dataset...")
    try:
        from ultralytics.data.utils import check_det_dataset
        data_dict = check_det_dataset('coco128.yaml')
        print(f"✅ Dataset ready at: {data_dict['path']}")
        print(f"   Classes: {data_dict['nc']}")
        print(f"   Train images: {data_dict['train']}")
        return True
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return False


if __name__ == "__main__":
    success = download_coco128()
    sys.exit(0 if success else 1)
