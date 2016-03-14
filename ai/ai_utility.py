
def convert_csv_to_floats(fname):
    with open(fname) as fh:
        text = fh.read()
    ## normalize line endings; split on lines
    rows = text.replace("\r","\n").replace("\n\n","\n").split("\n")
    ## "if row" excludes empty lines
    return [[float(v) for v in row.split(",")] for row in rows if row]

def normalize(data):
    nrows = len(data)
    ncolumns = len(data[0])
    ## sure, I could do these in double-list-comps, but it's less readable
    totals = [0] * ncolumns
    for row in data:
        for i in range(len(ncolumns)):
            totals[i] += row[i]
    averages = [totals[i] / nrows for i in range(nrows)]
    ## create new dataframe, same size as @a data, with normed values
    normed = [[0] * ncolumns for i in range(nrows)]
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            avg = averages[j]
            normed[i][j] = (value - avg) / avg
    return normed
