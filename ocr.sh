#!/usr/bin/env bash
echo "tesseract ./Images/sudoku_training.png stdout"
tesseract ./Images/sudoku_training.png stdout

echo "tesseract  -l eng ./Images/sudoku_training.png stdout"
tesseract ./Images/sudoku_training.png stdout -l eng

echo "tesseract -l eng --dpi 150 ./Images/sudoku_training.png stdout"
tesseract -l eng --dpi 150 ./Images/sudoku_training.png stdout

echo "tesseract -l eng nobatch digits ./Images/sudoku_training.png stdout "
tesseract -l eng nobatch digits ./Images/sudoku_training.png stdout