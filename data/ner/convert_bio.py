# Author: Karl Stratos (me@karlstratos.com)
#
# Input: a file of form ("BIO format")
#
#       John   B-PER
#       Smith  I-PER
#       works  O
#       at     O
#       New    B-ORG
#        ...
#
# Output: three files of form
#
#       [words] John Smith works at New ...
#       [spans] 0 1 4 6 ...
#       [entities] PER ORG ...
#
# Usage:
#         mkdir conll2003
#         for X in train dev test; do python convert_bio.py [data-path]/conll2003.${X} conll2003; done
import argparse
import os
from collections import Counter


def get_boundaries(bio):
    """
    Extracts an ordered list of boundaries. BIO label sequences can be either
    -     Raw BIO: B     I     I     O => {(0, 2, None)}
    - Labeled BIO: B-PER I-PER B-LOC O => {(0, 1, "PER"), (2, 2, "LOC")}
    """
    boundaries= []
    i = 0

    while i < len(bio):
        if bio[i][0] == 'O': i += 1
        else:
            s = i
            entity = bio[s][2:] if len(bio[s]) > 2 else None
            i += 1
            while i < len(bio) and bio[i][0] == 'I':
                if len(bio[i]) > 2 and bio[i][2:] != entity: break
                i += 1
            boundaries.append((s, i - 1, entity))

    return boundaries


def read_input_file(input_path):
    wseqs = []
    biolabelseqs = []
    with open(input_path) as infile:
        wseq = []
        biolabelseq = []
        for line in infile:
            toks = line.split()
            if toks:
                w, biotype = toks
                wseq.append(w)
                biolabelseq.append(biotype)
            else:
                if wseq: wseqs.append(wseq)
                if biolabelseq: biolabelseqs.append(biolabelseq)
                wseq = []
                biolabelseq = []
        if wseq: wseqs.append(wseq)
        if biolabelseq: biolabelseqs.append(biolabelseq)
    return wseqs, biolabelseqs


def extract_spans_entities(biolabelseqs, forbidden_entities={}):
    spanseqs = []
    entityseqs = []
    for biolabelseq in biolabelseqs:
        boundaries = get_boundaries(biolabelseq)
        spanseq = []
        entityseq = []
        for (i, j, entity) in boundaries:
            if not entity in forbidden_entities:
                spanseq.append((i, j))
                entityseq.append(entity)
        spanseqs.append(spanseq)
        entityseqs.append(entityseq)
    return spanseqs, entityseqs


def filter_empty(wseqs, spanseqs, entityseqs):
    wseqs_filtered = []
    spanseqs_filtered = []
    entityseqs_filtered = []
    for i in xrange(len(wseqs)):
        if len(spanseqs[i]) > 0:
            wseqs_filtered.append(wseqs[i])
            spanseqs_filtered.append(spanseqs[i])
            entityseqs_filtered.append(entityseqs[i])
    return wseqs_filtered, spanseqs_filtered, entityseqs_filtered


def report_stats(wseqs, spanseqs, entityseqs):
    entity_count = Counter()
    num_entities = 0.0
    length_entity_sum = 0.0
    for i in xrange(len(wseqs)):
        num_entities += len(spanseqs[i])
        for j in xrange(len(spanseqs[i])):
            entity_count[entityseqs[i][j]] += 1
            length_entity_sum += spanseqs[i][j][1] - spanseqs[i][j][0] + 1
    sorted_list = sorted(entity_count.items(), key=lambda x: x[1],
                         reverse=True)
    print "----------------------------------"
    print "# entity types: {0}".format(len(sorted_list))
    for entity, count in sorted_list:
        print "    {0:5s}   {1}".format(entity, count)
    print
    print "# sentences: {0}".format(len(wseqs))
    print
    print "# entities: {0}".format(int(num_entities))
    print "  average # of words in an entity: {0:.2f}".format(
        length_entity_sum / num_entities)
    print "  average # entities in a (non-empty) sentence: {0:.2f}".format(
        num_entities / len(wseqs))

def main(args):
    wseqs, biolabelseqs = read_input_file(args.infile)

    forbidden_entities = {entity: True for entity in args.forbid.split(',')} \
                         if args.forbid else {}
    spanseqs, entityseqs = extract_spans_entities(biolabelseqs,
                                                  forbidden_entities)

    wseqs, spanseqs, entityseqs = filter_empty(wseqs, spanseqs, entityseqs)
    report_stats(wseqs, spanseqs, entityseqs)

    _, filename = os.path.split(args.infile)
    if forbidden_entities: filename += "." + args.forbid

    with open(os.path.join(args.outdir, filename + ".words"), 'w') as wfile:
        for wseq in wseqs:
            wfile.write(' '.join(wseq) + '\n')

    with open(os.path.join(args.outdir, filename + ".spans"), 'w') as sfile:
        for spanseq in spanseqs:
            for (i, j) in spanseq:
                sfile.write(str(i) + ' ' + str(j) + ' ')
            sfile.write('\n')

    with open(os.path.join(args.outdir, filename + ".entities"), 'w') as efile:
        for entityseq in entityseqs:
            for entity in entityseq:
                efile.write(entity + ' ')
            efile.write('\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("infile", type=str, help="NER file in BIO format")
    argparser.add_argument("outdir", type=str, help="output directory")
    argparser.add_argument("--forbid", type=str,
                           help="forbidden entity types (separated by comma)")
    parsed_args = argparser.parse_args()
    main(parsed_args)
