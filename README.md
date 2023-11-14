<p align="center">
    <a href="https://github.com/anywherelan/awl/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/anywherelan/awl?color=brightgreen"></a>
    <a href="https://github.com/anywherelan/awl/releases"><img alt="GitHub release" src="https://img.shields.io/github/v/release/anywherelan/awl" /></a>
    <a href="https://github.com/anywherelan/awl/actions/workflows/test.yml"><img alt="Test build status" src="https://github.com/anywherelan/awl/actions/workflows/test.yml/badge.svg" /></a>
</p>


# About

Этот код выполняет обнаружение объектов на заданном изображении с использованием библиотеки MediaPipe.
Изображение загружается из файла, после чего применяется объект ObjectDetector для обнаружения объектов на изображении.
Результат обнаружения затем визуализируется путем отрисовки ограничительных рамок вокруг обнаруженных объектов и
отображения соответствующих меток и оценок. Визуализированное изображение сохраняется в новый файл.

# Setup

Рекомендуется использование [Anaconda](https://github.com/conda-forge/miniforge), после установки создайте среду python:
```bash
conda env create -f environment.yml
```



Для запуска видео, в параметр --cameraId передайте путь к файлу.