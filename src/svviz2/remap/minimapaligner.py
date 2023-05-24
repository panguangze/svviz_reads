import collections
import pysam
import mappy as mp
from svviz2.utility.misc import reverse_comp


class Aligner(object):
    def __init__(self, names_to_contigs):
        self.names_to_aligners = collections.OrderedDict()
        for name, contig in names_to_contigs.items():
            self.names_to_aligners[name] = mp.Aligner(
                seq = contig, preset="map-hifi")

    def align(self, seq):
        alns = []

        # revseq = reverse_comp(seq)
        # print(seq, "original sequence")

        for i, name in enumerate(self.names_to_aligners):
            aligner = self.names_to_aligners[name]
            falns = aligner.map(seq,cs=True, MD=True)
            # print(len(seq))
            for faln in falns:
                cur_aln = pysam.AlignedSegment()
                cur_aln.reference_id = i
                cur_aln.reference_start = faln.r_st
                cur_aln.set_tag("NM", faln.NM)
                cur_aln.cigarstring = faln.cigar_str
                cur_aln.is_reverse = True if faln.strand == '-1' or faln.strand == -1 else False
                if cur_aln.is_reverse:
                    cur_aln.query_sequence = reverse_comp(seq[faln.q_st:faln.q_en])
                else:
                    cur_aln.query_sequence = seq[faln.q_st:faln.q_en]
                alns.append((cur_aln, faln.q_st, faln.q_en, faln.ctg_len, faln.r_en))
        return alns