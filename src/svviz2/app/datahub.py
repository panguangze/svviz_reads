import collections
import logging
import os
import pysam
# import shutil
import tempfile

from svviz2.app import genomesource
from svviz2.app.sample import Sample
from svviz2.app import variants
from svviz2.io import getreads
from svviz2.io import vcfparser
from svviz2.io import read_filters
from svviz2.io import saverealignments
from svviz2.remap import maprealign
from svviz2.remap import genotyping
from svviz2.utility import misc
from svviz2.utility import intervals
import svviz2.debug as debug

logger = logging.getLogger(__name__)


def name_from_bam_path(bampath):
    return os.path.basename(bampath).replace(".bam", "").replace(".cram", "").replace(".sorted", "").replace(".sort",
                                                                                                             "").replace(
        ".", "_").replace("+", "_")


# def name_from_bed_path(bampath):
#     return os.path.basename(bampath).replace(".bed", "").replace(".sorted", "").replace(".sort", "").replace(".", "_").replace("+", "_").replace(".gz", "")


def get_internal_segments(sv, extend=20):
    chrom_parts = sv.chrom_parts("ref")

    internal_segments = []

    for part in chrom_parts:
        # print(part, "dddddd")
        for i, segment in enumerate(part.segments):
            if i == len(part.segments) - 1:
                internal_segment = intervals.Locus(segment.chrom, segment.start - extend, segment.start + extend, "+")
                internal_segments.append(internal_segment)
            elif i == 0:
                internal_segment = intervals.Locus(segment.chrom, segment.end - extend, segment.end + extend, "+")
                internal_segments.append(internal_segment)
            else:
                internal_segment = intervals.Locus(segment.chrom, segment.start - extend, segment.end + extend, "+")
                internal_segments.append(internal_segment)

    return internal_segments


def _get_pair_locus(aln1, aln2):
    assert aln1.reference_name == aln2.reference_name
    chrom = aln1.reference_name
    start = min(aln1.reference_start, aln2.reference_start)
    end = max(aln1.reference_end, aln2.reference_end)
    locus = intervals.Locus(chrom, start, end, "+")

    return locus


def _pair_passes(pair, segments):
    read1 = pair.original_read_ends["1"]
    read2 = pair.original_read_ends["2"]

    if not read1.is_proper_pair:
        return True
    for read in [read1, read2]:
        for pos in [0, -1]:
            if read.cigartuples[pos][0] in [4, 5] and read.cigartuples[pos][1] > 20:
                return False
    locus = _get_pair_locus(read1, read2)
    if intervals.overlaps(locus, segments):
        return True
    return False


def filter_pair_batch(batch, variant):
    passing = []
    segments = get_internal_segments(variant)

    for pair in batch:
        if _pair_passes(pair, segments):
            passing.append(pair)
    return passing


class DataHub(object):
    def __init__(self):
        self.args = None
        self.align_distance = 0
        self.samples = collections.OrderedDict()
        self.genome = None

        self.alleleTracks = collections.defaultdict(collections.OrderedDict)
        # self.annotationSets = collections.OrderedDict()

        self.aligner_type = "bwa"

        self.should_genotype = True

        self.should_render = True
        self.should_generate_reports = True
        self.should_generate_dotplots = True

        self.temp_dir = None
        self.prepar_fa = None

    def prepar_input(self):
        for sample_name, sample in self.samples.items():
            logger.info("Analyzing sample {}".format(sample_name))
            for batch in getreads.get_read_batch(sample, self):
                if sample.single_ended:
                    logger.info("parser {} reads".format(len(batch)))
                else:
                    logger.info("parser {} read pairs".format(len(batch)))
                # print(batch)
                # break
                sample.parse_bam(batch)
                # try:
                #     sample.bam_sort_index(os.path.join(sample.datahub.args.outdir, ".".join([sample.name, "prepar", "bam"])))
                # except:
                #     print("ERROR!" * 30)
                #     raise

            # sample.finish_writing_realignments()

    def genotype_cur_variant(self):
        for sample_name, sample in self.samples.items():
            logger.info("Analyzing sample {}".format(sample_name))
            ref_count = 0
            alt_count = 0

            for batch in getreads.get_read_batch(sample, self):
                if sample.single_ended:
                    logger.info("Analyzing {} reads".format(len(batch)))
                else:
                    logger.info("Analyzing {} read pairs".format(len(batch)))

                    if self.args.fast:
                        print("BEFORE::", len(batch))
                        # batch = filter_pair_batch(batch, self.variant)
                        # print("AFTER:: ", len(batch))
                # print(batch)
                # break

                diff_len = abs(len(self.variant.seqs("ref")) - len(self.variant.seqs("alt")))
                if diff_len == 0:
                    if len(self.variant.segments("ref")) == 3:
                        diff_len = len(self.variant.segments("ref")[1])
                    elif len(self.variant.segments("alt")) == 3:
                        diff_len = len(self.variant.segments("alt")[1])
                    else:
                        diff_len = 500
                print(self.variant.segments("ref"), "segments")
                # print(self.variant.segments("alt"), "segments")
                # print(diff_len, "diff_len")
                aln_sets = maprealign.map_realign(batch, self, sample, diff_len)

                cur_ref_count, cur_alt_count = genotyping.assign_reads_to_alleles(
                    aln_sets,
                    variants.get_breakpoints_on_local_reference(self.variant, "ref"),
                    variants.get_breakpoints_on_local_reference(self.variant, "alt"),
                    sample.read_statistics, diff_len, self.args.percent)
                ref_count += cur_ref_count
                alt_count += cur_alt_count

                sample.add_realignments(aln_sets)

            print("REF:", ref_count, "ALT:", alt_count)

            sample.finish_writing_realignments()

    def __getstate__(self):
        """ allows pickling of DataHub()s """
        state = self.__dict__.copy()
        # del state["args"]
        del state["genome"]
        return state

    def cleanup(self):
        pass
        # temp_dir = os.path.join(self.args.outdir, "svviz2-temp")
        # if os.path.exists(temp_dir):
        #    shutil.rmtree(temp_dir)
        # self.temp_dir.cleanup()

    def get_variants(self):
        vcf = vcfparser.VCFParser(self)

        good_variants = 0

        for count, variant in enumerate(vcf.get_variants()):
            print(count, "count")
            if variant is None:
                continue
            if self.args.first_variant is not None and count < self.args.first_variant:
                continue
            if self.args.last_variant is not None and count > self.args.last_variant:
                continue

            logger.info("Working on variant {}: {}".format(count, variant))

            # for allele in ["alt", "ref"]:
            #     print(f"{allele }"*10)
            #     print(variant.segments(allele))
            #     # for part, seq in variant.seqs(allele).items():
            #         # print(f"{allele }"*10)
            #         # print(seq)
            good_variants += 1
            self.set_cur_variant(variant)
            yield variant

        if good_variants == 0:
            logger.error("Found no variants in file -- is it formatted properly?")

    def set_cur_variant(self, variant):
        self.variant = variant
        for sample in self.samples.values():
            if sample.has_resolved():
                self.should_genotype = False
                logger.info(
                    "Found realignments".format(self.variant))
                return

        local_coords_in_full_genome = self.variant.search_regions(self.align_distance)
        self.genome.blacklist = local_coords_in_full_genome

        self.local_ref_genome_source = genomesource.GenomeSource(
            self.variant.seqs("ref"), aligner_type=self.aligner_type)
        self.local_alt_genome_source = genomesource.GenomeSource(
            self.variant.seqs("alt"), aligner_type=self.aligner_type)

        # TODO: if savereads, then save to a proper file, otherwise save to temporary space
        # ... or just always save

        if self.args.savereads or self.args.render_only:
            outdir = self.args.outdir
        else:
            # outdir = os.path.join(self.args.outdir, "svviz2-temp")
            outdir = self.temp_dir.name
            misc.ensure_dir(outdir)

        for sample_name, sample in self.samples.items():
            for allele in ["ref", "alt", "amb"]:
                sample.outbam_paths[allele] = os.path.join(
                    outdir,
                    ".".join([variant.short_name(), sample_name, allele, "bam"]))

            # sample.outbam_paths["ref"] = os.path.join(
            #     self.args.outdir, "{}.{}.ref.bam".format(variant.short_name(), sample_name))

        if self.args.savereads:
            if self.prepar_fa == None:
                self.prepar_fa = open(os.path.join(self.args.outdir, "ref.alt.prepar.fa"), "w")
            for allele in ["alt", "ref"]:
                # outpath = os.path.join(self.args.outdir, "{}.genome.{}.fa".format(variant.short_name(), allele))
                # with open(self.prepar_fa, "w") as genome_file:
                for name, seq in self.variant.seqs(allele).items():
                    self.prepar_fa.write(">{}-{}\n{}\n".format(variant.name, allele, seq))

        self.should_genotype = False
        logger.info("Looking for existing realignments; will try to open a few bam files that may not yet exist...")
        for sample in self.samples.values():
            if not sample.has_resolved():
                self.should_genotype = True
        if not self.should_genotype:
            logger.info("Found realignments for all samples for event {}; skipping realignment".format(self.variant))

    def set_args(self, args):
        debug.IS_DEBUG = args.debug
        EXTRA_ARG_TYPES = ["single_ended", "sequencer", "split_hap", "max_base_quality"]
        self.args = args

        self.aligner_type = args.aligner
        assert self.aligner_type in ["bwa", "ssw", "minimap"]

        self.genome = genomesource.FastaGenomeSource(args.ref)

        if self.args.outdir is None:
            self.args.outdir = os.getcwd()
        misc.ensure_dir(self.args.outdir)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="svviz2-temp", dir=self.args.outdir)

        for bam_description in self.args.bam:
            fields = bam_description.split(",")

            bam_path = fields.pop(0)
            name = name_from_bam_path(bam_path)

            # get unique name by appending _i as needed
            i = 0
            while name in self.samples:
                i += 1
                curname = "{}_{}".format(name, i)
                if curname not in self.samples:
                    name = curname
                    break

            extra_args = {}
            for field in fields:
                if field.count("=") != 1:
                    message = "Error specifying additional sample arguments: field '{}' should" \
                              "be of format 'key=value'"
                    raise Exception(message.format(field))
                key, value = field.split("=")
                if not key in EXTRA_ARG_TYPES:
                    message = "Error specifying additional sample arguments: key '{}' must be" \
                              "one of {}"
                    raise Exception(message.format(key, ",".join(EXTRA_ARG_TYPES)))
                extra_args[key] = value

            split_hap = False
            if "split_hap" in extra_args:
                split_hap = misc.str_to_bool(extra_args["split_hap"])
            if split_hap:
                for hap in [1, 2, None]:
                    hap_name = "{}_hap{}".format(name, hap)
                    if hap is None:
                        hap_name = "{}_hap-unknown".format(name)
                    sample = Sample(hap_name, bam_path, self, extra_args)
                    sample.read_filter = read_filters.get_haplotype_filter(hap)
                    self.samples[hap_name] = sample
            else:
                sample = Sample(name, bam_path, self, extra_args)
                self.samples[name] = sample

        if self.args.no_render:
            self.should_render = False
        if self.args.no_dotplots:
            self.should_generate_dotplots = False
        if self.args.no_report:
            self.should_generate_reports = False

        if self.args.render_only:
            self.should_generate_dotplots = False
            self.should_generate_reports = False

        if self.args.dotplots_only:
            self.should_render = False
            self.should_generate_reports = False

        if self.args.report_only:
            self.should_render = False
            self.should_generate_dotplots = False

        # if not (self.should_render and self.should_generate_dotplots and self.should_generate_reports):
        # self.should_genotype = False
        # if self.args.annotations:
        #     for annoPath in self.args.annotations:
        #         name = nameFromBedPath(annoPath)
        #         if annoPath.endswith(".bed") or annoPath.endswith(".bed.gz"):
        #             self.annotationSets[name] = annotations.AnnotationSet(annoPath)
        #         else:
        #             if not (annoPath.endswith(".gff") or annoPath.endswith(".gff.gz") \
        #                 or annoPath.endswith(".gtf") or annoPath.endswith(".gtf.gz")):
        #                 logging.warn("Unknown annotation file extension; trying to parse as if GTF/GFF format: '{}'".format(annoPath))
        #             self.annotationSets[name] = gff.GeneAnnotationSet(annoPath)

    def __iter__(self):
        return iter(list(self.samples.values()))

