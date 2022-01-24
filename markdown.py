def markdown(df):
    i=0
    txt="|"
    for col in df.columns:
        txt=txt+"{}|".format(col)
    txt+="\n"
    txt+="|"
    for col in df.columns:
        txt=txt+"{}|".format("-------------")
    txt+="\n"
    for row in df.index:
        for col in df.columns:
            if col!="step":
                txt=txt+'{:0.2e}'.format(df.loc[row,col])+"|"
            else:
                txt=txt+'{:d}'.format(df.loc[row,col])+"|"
        txt+="\n"
    return txt
