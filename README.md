# PDF2GCODE

## Dependencies

* tracer from schraffer - https://github.com/hacxman/schraffer.git
* pdftoppm - usually found in poppler-utils package (on Fedora)

## Installing and configuration

```bash
$ git clone https://github.com/hacxman/pdf2gcode.git
```

Clone schraffer repo and copy tracer to your bin directory (for example ~/bin).
Install pdftoppm (poppler-utils on Fedora and Debian).

Edit settings.json according to your machine specs.

### Using pdf_to_gcode

Using pdf_to_gcode.py
```bash
$ ./pdf_to_gcode.py your_pdf.pdf
```
Produces your_pdf.pdf.ngc and your_pdf.pdf.levelling.ngc

### Levelling

Pdf_to_gcode produces levelling script, run it on your machine,
and make sure you have your touch probe installed (otherwise you'll
break your machine). That should produce probe-results.txt. Copy it
back to directory with pdf2gcode.

Run `./levelgcode.py X Y your_pdf.pdf.ngc`
where X and Y are irrelevant at the moment :D.
That will automatically use scale_gcode and level your .ngc according to
obtained levelling data. Produces your_pdf.pdf.ngc.levelled.ngc.

Enjoy and please fill issues.
