import collections
import logging
# import numpy
import pyfaidx
import seqlib
import sys
import traceback

from svviz2.utility import intervals, misc
# from svviz2.remap import mapq
#from svviz2.remap import ssw_aligner
from svviz2.remap import minimapaligner
from svviz2.remap.alignment import Alignment
import svviz2.debug as debug

logger = logging.getLogger(__name__)

try:
    from ..remap import _mapq
except:
    logger.error(traceback.format_exc())
    logger.error("ERROR: Failed to import mapq module; this is almost certainly due to "
                 "a version mismatch between the installed versions of svviz2 and pysam;"
                 "try reinstalling svviz2")
    sys.exit()



PARAMS = {
    "illumina":{
        "min_seed_length":19,
        "min_chain_weight":0,
        "gap_open":6,
        "gap_extension":1,
        "mismatch_penalty":4,
        "3prime_clipping_penalty":5,
        "5prime_clipping_penalty":5,
        "reseed_trigger":1.5,
        },
    "pacbio": {
        "min_seed_length":17,
        "min_chain_weight":40,
        "gap_open":1,
        "gap_extension":1,
        "mismatch_penalty":1,
        "3prime_clipping_penalty":0,
        "5prime_clipping_penalty":0,
        "reseed_trigger":10,
    },
    "nanopore": {
        "min_seed_length":14,
        "min_chain_weight":20,
        "gap_open":1,
        "gap_extension":1,
        "mismatch_penalty":1,
        "3prime_clipping_penalty":0,
        "5prime_clipping_penalty":0,
        "reseed_trigger":10,
    }
}

## These are pretty much identical to the versions in genomeview, but each method
## adds some functionality to deal with bwa/ssw alignment

class GenomeSource:
    def __init__(self, names_to_contigs, aligner_type="bwa"):
        self.names_to_contigs = collections.OrderedDict(names_to_contigs)
        self._bwa = None
        self._ssw = None
        self._minimap = None
        self._blacklist = None

        self.aligner_type = aligner_type
        self.max_base_quality = 40.0
        # self.max_base_quality = 93

    def get_seq(self, chrom, start, end, strand):
        seq = self.names_to_contigs[chrom][start:end+1]
        if strand == "-":
            seq = misc.reverse_comp(seq)
        return seq

    def keys(self):
        return list(self.names_to_contigs.keys())

    @property
    def blacklist(self):
        return self._blacklist

    @blacklist.setter
    def blacklist(self, blacklist_loci):
        self._blacklist = []

        for locus in blacklist_loci:
            cur_chrom = misc.match_chrom_format(locus.chrom, list(self.keys()))
            self._blacklist.append(intervals.Locus(cur_chrom, locus.start, locus.end, locus.strand))

    def align(self, read, diff_len):
        alns = []
        if self.aligner_type == "minimap2":
            qualities = read.query_qualities
        else:
            qualities = read.original_qualities()

        # TODO
        # print(read.query_name, "read.query_name")
        raw_alns = self.aligner.align(read.original_sequence())
        # raw_alns = [read._read]
        # raw_alns = read

        for aln in raw_alns:
            if self.aligner_type == "minimap2":
                aln = Alignment(aln[0], aln[1], aln[2], aln[3], aln[4], "minimap2")
                if aln.is_reverse and qualities is not None:
                    # print(aln._read.query_qualities, "qualities1x")
                    # print(len(qualities), "qualities2x")
                    # print(read.query_name, aln.q_st, aln.q_en, "q_st, q_en")
                    aln._read.query_qualities = qualities[int(aln.q_st):int(aln.q_en)][::-1]
                else:
                    # print(aln._read.query_qualities, "qualities1")
                    # print(len(qualities), "qualities2")
                    aln._read.query_qualities = qualities[int(aln.q_st):int(aln.q_en)]
            else:
                aln = Alignment(aln, "bwa")
                if aln.is_reverse and qualities is not None:
                    aln._read.query_qualities = qualities[::-1]
                else:
                    aln._read.query_qualities = qualities

            aln.chrom = self.keys()[aln.reference_id]
            aln._read.query_name = read.query_name
            aln.original_seq_len = len(read.original_sequence())

            # if self.blacklist is not None:
                # print("....", aln.locus, self.blacklist, misc.overlaps(aln.locus, self.blacklist))
            if self.blacklist is None or not intervals.overlaps(aln.locus, self.blacklist):
                aln.source = self
                aln.chrom = self.keys()[aln.reference_id]
                self.score_alignment(aln, diff_len)
                aln.set_tag("mq", read.mapq)
                alns.append(aln)
    
        return alns

    def score_alignment(self, aln, diff_len):
        # TODO: move the mapqcalculator code to here

        # mc = mapq.MAPQCalculator(self)
        # aln.score = mc.get_alignment_end_score(aln)

        ref_seq = self.get_seq(aln.chrom, aln.reference_start, aln.reference_end, "+").upper()
        aln.score = _mapq.get_alignment_end_score(aln._read, ref_seq, max_quality=self.max_base_quality)
        if self.aligner_type == "minimap2":
            matched_count,read_count = _mapq.diff_region_similarity(aln._read, aln.q_st, aln.q_en, ref_seq, diff_len, max_quality=self.max_base_quality)
        # matched_count,read_count = result_str.split("-")
        aln.matched_count = int(matched_count)
        aln.read_count_in_region = int(read_count)

        # print(aln._read.query_sequence, "query_sequence")
        # print(ref_seq, "ref_sequence")
        # print(aln.cigarstring, aln._read.get_tag("NM"), "cigar")
        # print(aln.q_st, aln.q_en, "q_st, q_en")
        # print(aln._read.reference_start, aln._read.reference_end, "t_st, t_en")
        # print(aln.cigarstring, aln._read.get_tag("NM"), "cigar", aln.score,aln._read.query_name, "aln.score")
        # aln.score = s2

        # assert numpy.isclose(aln.score, s2, rtol=1e-5), "{} :: {}".format(aln.score, s2)

    @property
    def aligner(self):
        if self.aligner_type == "bwa":
            return self.bwa
        elif self.aligner_type == "ssw":
            return self.ssw
        elif self.aligner_type == "minimap2":
            return self.minimap

    @property
    def ssw(self):
        if self._ssw is None:
            self._ssw = ssw_aligner.Aligner(self.names_to_contigs)
        return self._ssw

    @property
    def minimap(self):
        if self._minimap is None:
            self._minimap = minimapaligner.Aligner(self.names_to_contigs)
        return self._minimap
    @property
    def bwa(self):
        """
        pacbio: -k17 -W40 -r10 -A1 -B1 -O1 -E1 -L0  (PacBio reads to ref)
        ont2d: -k14 -W20 -r10 -A1 -B1 -O1 -E1 -L0  (Oxford Nanopore 2D-reads to ref)
        """
        if self._bwa is None:
            self._bwa = seqlib.BWAWrapper()
            self._bwa.makeIndex(self.names_to_contigs)

        return self._bwa

    def set_aligner_params(self, sequencer, max_base_quality=40.0):
        if self.aligner_type != "bwa":
            print("not bwa... skipping setting aligner settings")
            return

        params = PARAMS[sequencer]
        if "min_seed_length" in params:
            self.bwa.SetMinSeedLength(params["min_seed_length"])
        if "min_chain_weight" in params:
            self.bwa.SetMinChainWeight(params["min_chain_weight"])

        if "mismatch_penalty" in params:
            self.bwa.SetMismatchPenalty(params["mismatch_penalty"])

        if "gap_open" in params:
            self.bwa.SetGapOpen(params["gap_open"])
        if "gap_extension" in params:
            self.bwa.SetGapExtension(params["gap_extension"])

        if "3prime_clipping_penalty" in params:
            self.bwa.Set3primeClippingPenalty(params["3prime_clipping_penalty"])
        if "5prime_clipping_penalty" in params:
            self.bwa.Set5primeClippingPenalty(params["5prime_clipping_penalty"])

        if "reseed_trigger" in params:
            self.bwa.SetReseedTrigger(params["reseed_trigger"])

        self.max_base_quality = max_base_quality

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_bwa"] = None
        return state


class FastaGenomeSource(GenomeSource):
    """ pickle-able wrapper for pyfaidx.Fasta """
    def __init__(self, path, aligner_type="bwa"):
        self.path = path
        self._fasta = None
        self._bwa = None
        self._blacklist = None
        self.aligner_type = aligner_type
        
    def get_seq(self, chrom, start, end, strand):
        chrom = misc.match_chrom_format(chrom, list(self.fasta.keys()))

        seq = self.fasta[chrom][start:end+1]
        if strand == "-":
            seq = misc.reverse_comp(seq)
        return seq

    def keys(self):
        return list(self.fasta.keys())

    @property
    def fasta(self):
        if self._fasta is None:
            self._fasta = pyfaidx.Fasta(self.path, as_raw=True)
        return self._fasta

    @property
    def bwa(self):
        if self._bwa is None:
            logger.info("Loading bwa index from file {}...".format(self.path))
            
            self._bwa = seqlib.BWAWrapper()
            result = self._bwa.loadIndex(self.path)

            if not result:
                raise IOError("Failed to load bwa index from file {}".format(self.path))
            
            logger.info("Loading bwa index done.")

        return self._bwa

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fasta"] = None
        state["_bwa"] = None
        return state
