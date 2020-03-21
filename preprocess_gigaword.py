import os
import re
import glob
import codecs
import string
from optparse import OptionParser


# Preprocess a gigaword corpus to extract the text of each article, remove punctuation, and write it all
# to one file with one article per line


replace = re.compile('[%s]' % re.escape(string.punctuation))


def main():
    usage = "%prog input_dir output_file"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    output_file = args[1]

    with codecs.open(output_file, 'w') as f:
        f.write('')

    files = glob.glob(os.path.join(input_dir, 'nyt_eng_200*'))
    files.sort()

    count = 0
    for f in files:
        print(f)
        lines = []
        doc = ''
        text = ''
        read = False
        with codecs.open(f, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('<DOC id'):
                    # start a new document
                    text = ''
                    if count % 1000 == 0 and count > 0:
                        print(count)
                elif line == '<P>':
                    pass
                elif line == '</P>':
                    pass
                elif line == '<TEXT>':
                    # start reading
                    read = True
                elif line == '</TEXT>':
                    # stop reading and save document
                    read = False

                    # remove single quotes (to simplify contractions to a single word) as well as @ and . and :
                    text = clean_text(text)

                    # a few documents only have headlines and no text
                    if len(text) > 0:
                        lines.append(text)
                        count += 1

                    text = ''
                elif read:
                    if line == '<HEADLINE>' or line == '</HEADLINE>' or line == '<DATELINE>' or line == '</DATELINE>':
                        print("Unexpectedly encountered headline/dateline tag")
                    # add text to line
                    text += line + ' '

        print("Adding articles to file")
        with codecs.open(output_file, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    print("%d documents" % count)


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', '', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # replace single quotes with underscores
    text = re.sub(r'\'', '_', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation with spaces
    text = replace.sub(' ', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main()
