import typst
try:
    typst.compile("test_typ.typ", output="test.pdf")
except Exception as e:
    print(e)
