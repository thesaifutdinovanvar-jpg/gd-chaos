# Градиентный спуск при больших шагах: хаос и фрактальная область сходимости

Курсовая работа, 1 курс

**Автор:** Сайфутдинов Анвар Ирекович, группа 25.Б21-мм  
**Научный руководитель:** д.ф.-м.н., профессор Мокаев Т.Н.  
**Кафедра:** прикладной кибернетики, СПбГУ

## Описание

Исследование хаотической динамики градиентного спуска при больших learning rate:

- Скалярная и матричная факторизация
- Визуализация бассейнов сходимости
- Оценка фрактальной размерности (box-counting)
- Сравнение с теоретическим критическим шагом η*
- Анализ чувствительности к начальным условиям
- Связь с edge of stability и catapult mechanism

## Файлы

- `code/experiments.py` - код всех экспериментов
- `figures/` - графики (17 файлов)
- `main.tex`, `biblio.bib` - LaTeX исходники
- `main.pdf` - готовый отчёт (17 страниц)

## Запуск

```bash
pip install numpy matplotlib

cd code
python experiments.py
```

## Компиляция LaTeX

```bash
pdflatex main.tex
biber main
pdflatex main.tex
```

## Литература

1. Liang, Montúfar. Gradient Descent with Large Step Sizes. arXiv:2509.25351, 2025
2. McDonald et al. Fractal basin boundaries. Physica D, 1985
3. Cohen et al. Edge of Stability. ICLR, 2021
4. Lewkowycz et al. Catapult mechanism. arXiv:2003.02218, 2020
