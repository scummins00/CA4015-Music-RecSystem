# CA4015-Music-RecSystem
This repository hosts the Jupyter Book and accompanying files used for CA4015 Assignment 3 &amp; 4 - Music Recommender System and Neural Networks.

## Required Packages
Please find all the packages required to run this book as intended in the `requirements.txt` file.

## Building the Jupyter Book:
1. Make changes to your book in the `main` branch.
2. Rebuild the book with `jupyter-book build book/`
3. Use `ghp-import -n -p -f book/_build/html` to push the new pages to the gh-pages branch
4. Access the Book [here](https://scummins00.github.io/CA4015-Music-RecSystem/intro.html).

## Building a PDF
This Jupyter Book is available in pdf version. To create the pdf, you will need `pyppeteer` to convert HTML to PDF.

Install `pyppeteer` using the following command:
`pip install pyppeteer`

Once installed, you can build the PDF using:
`jupyter-book build book/ --builder pdfhtml`

Please find the book at: `book/_build/pdf/book`