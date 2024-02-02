

def by_label(ats, info_label):
    if info_label == None:
        return {"no_label":ats}
    data = {}
    for at in ats:
        if info_label not in at.info.keys():
            label = "no_label"
        else:
            label = at.info[info_label]

        if label not in data.keys():
            data[label] = []
        data[label].append(at)

    return data

