# Plate Project: YOLO -> Crop -> GAN Restore -> OCR -> Evaluation

Bu repo su sirayi izler:

1. Arac goruntulerinden YOLO ile plaka tespiti
2. Tespit edilen bolgeden plaka crop alma
3. Crop'lardan sentetik karli / bozulmus veri uretme
4. GAN ile plaka restorasyonu
5. Restorasyon oncesi ve sonrasi OCR
6. Ground truth varsa before/after degerlendirme

## Onemli dosyalar

- `src/train_yolo.py`: YOLO egitimi
- `src/detect_and_crop.py`: YOLO detection + crop
- `src/generate_snow_data.py`: paired synthetic degradation uretimi
- `src/gan_restoration.py`: generator inference modulu
- `src/ocr_utils.py`: OCR varyantlari ve plaka aday secimi
- `src/run_ocr.py`: GAN destekli OCR calistirma
- `src/evaluate_pipeline.py`: OCR before/after metrikleri
- `src/inference_pipeline.py`: tek gorsel icin full inference zinciri
- `src/run_full_pipeline.py`: uctan uca tum sira

## Config

- `configs/yolo_config.yaml`
- `configs/pipeline_config.yaml`

`pipeline_config.yaml` icinde GAN agirligi varsayilan olarak:

- `C:/Users/Nidasu/Downloads/best_generator_by_ssim.pth`

OCR icin bu tercih edildi cunku karakter formunu koruma acisindan `best_generator_by_loss.pth` yerine daha guvenli bir varsayim.

## Kurulum

```bash
cd plate_project
pip install -r requirements.txt
python src/prepare_folders.py
```

## OCR Calistirma

```bash
python src/run_ocr.py \
  --input_csv data/synthetic_snow_pairs/pairs.csv \
  --config configs/pipeline_config.yaml
```

Uretilen dosyalar:

- `outputs/ocr_results/ocr_results.csv`
- `outputs/ocr_results/restored/*.png`

`ocr_results.csv` icinde sunlar tutulur:

- `prediction_before`
- `prediction_after`
- `raw_before_candidates`
- `raw_after_candidates`

## Tek gorselde full pipeline

```bash
python src/inference_pipeline.py --image data/test_samples/example.jpg --config configs/pipeline_config.yaml
```

Ciktilar:

- `outputs/detections/*_det.jpg`
- `outputs/crops/*_crop.png`
- `outputs/ocr_results/single_restored/*_restored.png`
- `outputs/ocr_results/*_pipeline_result.txt`

## Degerlendirme

```bash
python src/evaluate_pipeline.py \
  --results outputs/ocr_results/ocr_results.csv \
  --pairs data/synthetic_snow_pairs/pairs.csv
```

Ground truth `plate_text` doluysa:

- `exact_match_before`
- `exact_match_after`
- `char_acc_before`
- `char_acc_after`

olculur. `plate_text` bos ise script bunu acikca yazar ve yalanci sifir metrik uretmez.
