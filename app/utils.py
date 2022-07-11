import pandas as pd

ALLOWED_EXTENSIONS = {'csv', 'wav'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Inspired by https://stackoverflow.com/q/3930713/12946736
def print_items(dictObj, depth=0):
    p=[]
    p.append('<ul style="list-style: none;">\n')
    for k,v in dictObj.items():
        k = str(k)
        if isinstance(v, dict):
            p.append('<li>'+ k+ ':')
            p.append(print_items(v, depth+1))
            p.append('</li>')
        else:
            p.append('<li>'+ k+ ': '+ v+ '</li>')
    p.append('</ul>' + ('<br>' if depth == 1 else '') + '\n')
    return '\n'.join(p)


def prepare_data(files):
    for file in files:
        if not allowed_file(file.filename):
            raise ValueError(f"The extension for {file.filename} is not allowed.")

    text_files, audio_files = text_audio = ([], [])

    for file in files:
        text_audio[file.filename.rsplit('.', 1)[1].lower() == "wav"].append(file)

    text_files_texts = []

    for file in text_files:
        csv = pd.read_csv(file, header=None, na_filter=False, dtype=str).to_numpy()

        if csv.shape[1] > 1:
            raise ValueError(f"{file.filename} has more than 1 column.")

        rows = csv.flatten()
        text_files_texts.append(rows)

    return audio_files, (text_files, text_files_texts)
