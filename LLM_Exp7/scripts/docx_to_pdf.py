"""Convert a .docx to PDF through Word COM, then render pages to PNG."""
import os
import sys

import win32com.client as win32


def to_pdf(docx_path, pdf_path):
    docx_path = os.path.abspath(docx_path)
    pdf_path = os.path.abspath(pdf_path)
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(docx_path, ReadOnly=True)
    try:
        doc.ExportAsFixedFormat(pdf_path, 17)  # wdExportFormatPDF
    finally:
        doc.Close(False)
        word.Quit()
    return pdf_path


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    print(to_pdf(src, dst))
