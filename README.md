# Restoration of Circuits

Пайплайн для восстановления схем из сжатых изображений с использованием ClearML.

## Структура пайплайна

1. **data_preparation** — создаёт пары LQ/HQ изображения и делит на train/val/test
2. **preprocessing** — resize и нормализация изображений
3. **train_model** — обучение Conv AE (U-Net архитектуры)
4. **inference** — применение модели на тестовых данных
5. **postprocess** — бинаризация и морфологическая обработка
6. **vectorize** — экспорт в SVG/DXF для CAD-систем
7. **evaluate** — расчёт метрик (PSNR, SSIM, LPIPS, OCR)
8. **pipeline_controller** — управляет запуском всех задач

## Установка

```bash
uv sync
```

## Запуск

```bash
python clearml_pipeline_unified.py
```

Все задачи и артефакты отслеживаются в ClearML UI.
