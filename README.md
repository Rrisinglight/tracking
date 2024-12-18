# Object Detection with MediaPipe

## About

Этот проект выполняет обнаружение объектов с использованием **MediaPipe**.  
Результаты визуализируются с метками и оценками, а итоговое изображение сохраняется в файл.

### Задача и цели:

- Разработка программно-аппаратной системы для отслеживания БПЛА.
- Использование поворотного устройства для автоматической корректировки направленных антенн.  
- Обеспечение стабильной связи, стриминга видеосигнала и обнаружения объектов в реальном времени.  
- Тестирование производительности **Raspberry Pi 5** для обработки видеосигнала и оценки её потенциала в будущих задачах.  

### Применённые методы ИИ:

- **TensorFlow Lite**:  
  Оптимизация модели для устройств с ограниченными ресурсами.  

- **Квантование до int8**:  
  Уменьшение размера модели и ускорение работы.  

- **Оптимизация графа вычислений**:  
  Упрощение операций для эффективной работы на ARM-архитектуре.  

- **OpenCV**:  
  Улучшение качества видео и наложение фильтров в реальном времени.  

Решение интегрировано с **Raspberry Pi 5**. Использован открытый датасет. Главной целью было не только показать качество модели, но и протестировать возможности платформы для обработки видеосигнала и обнаружения объектов.

---

## Setup

Для настройки среды рекомендуется использовать [Anaconda](https://github.com/conda-forge/miniforge).  

1. Установите Anaconda.  
2. Создайте среду Python:  
   ```bash
   conda env create -f environment.yml
   ```
3. Активируйте среду:  
   ```bash
   conda activate <environment_name>
   ```