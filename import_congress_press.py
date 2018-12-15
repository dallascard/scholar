import os
import re
import sys
import glob
import json
from optparse import OptionParser

import pandas as pd


def main(args):
    usage = "%prog path/to/CongressPressExpand output_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--a', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args(args)

    input_dir = args[0]
    output_dir = args[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sanders_files = glob.glob(os.path.join(input_dir, 'Sanders', '*'))
    obama_files = glob.glob(os.path.join(input_dir, 'Obama', '*'))
    klobuchar_files = glob.glob(os.path.join(input_dir, 'Klobuchar', '*'))
    mccain_files = glob.glob(os.path.join(input_dir, 'McCain', '*'))
    graham_files = glob.glob(os.path.join(input_dir, 'Graham', '*'))
    coburn_files = glob.glob(os.path.join(input_dir, 'Coburn', '*'))

    outlines = load_files(sanders_files, 'Sanders', 'D', -0.7170)
    outlines += load_files(obama_files, 'Obama', 'D', -0.3910)
    outlines += load_files(klobuchar_files, 'Klobuchar', 'D', -0.2770)
    outlines += load_files(mccain_files, 'McCain', 'R', 0.3820)
    outlines += load_files(graham_files, 'Graham', 'R', 0.4270)
    outlines += load_files(coburn_files, 'Coburn', 'R', 0.8580)

    print("Saving files to", output_dir)
    with open(os.path.join(output_dir, 'train.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    n_lines = len(outlines)
    score_df = pd.DataFrame([line['score'] for line in outlines], index=range(n_lines), columns=['score'])
    score_df.to_csv(os.path.join(output_dir, 'train.score.csv'))
    print("Done!")


def load_files(files, senator, party, score):
    outlines = []
    print("{:d} files from {:s}".format(len(files), senator))
    files.sort()
    for infile in files:
        with open(infile, encoding='Windows-1252') as f:
            name = os.path.basename(infile)
            match = re.match(r'(\d+)([a-zA-Z]+)(\d+)', name)
            day = match.group(1)
            month = match.group(2)
            year = int(match.group(3))
            date = day + month + str(year)
            text = f.read()
            text = re.sub(r'\s', ' ', text)
            outlines.append({'id': name, 'text': text, 'senator': senator, 'date': date, 'year': year, 'month': month, 'party': party, 'score': score})
    return outlines


if __name__ == '__main__':
    main(sys.argv[1:])
