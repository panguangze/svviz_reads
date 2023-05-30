import math

import numpy

import json
from svviz2.utility import intervals, statistics
import svviz2.debug as debug



def calculate_genotype_likelihoods(ref, alt, priors=[0.05, 0.5, 0.95], max_qual=200):
    """
    calculates the bayesian genotype likelihoods as per Chiang et al (2015)
    """

    ref = int(ref)
    alt = int(alt)

    log_combo = statistics.log_choose(ref+alt, alt)

    log_prob_homref = log_combo + alt * numpy.log10(priors[0]) + ref * numpy.log10(1-priors[0])
    log_prob_het    = log_combo + alt * numpy.log10(priors[1]) + ref * numpy.log10(1-priors[1])
    log_prob_homalt = log_combo + alt * numpy.log10(priors[2]) + ref * numpy.log10(1-priors[2])

    # This is the "genotype likelihoods", aka GL
    log_probs = numpy.array([log_prob_homref, log_prob_het, log_prob_homalt])
    
    log_prob_sum = numpy.log10((10**log_probs).sum())
    genotype_qualities = 1-(10**log_probs/10**log_prob_sum)
    # print("::", genotype_qualities)
    with numpy.errstate(divide="ignore"):
        phred_genotype_qualities = numpy.abs(-10 * numpy.log10(genotype_qualities))
    phred_genotype_qualities[phred_genotype_qualities>max_qual] = max_qual
    return log_probs, phred_genotype_qualities


def get_overlaps(read_locus, unsequenced_insert_locus, breakpoints):
    overlaps = {}

    for breakpoint in breakpoints:
        if not read_locus.overlapsAnysense(breakpoint):
            continue
        if len(breakpoint) > 1:
            raise NotImplementedError("breakpoints with size > 1")

        cur_overlap = min([
            breakpoint.start - read_locus.start,
            read_locus.end - breakpoint.start])

        extension = read_locus.end - breakpoint.start

        if unsequenced_insert_locus:
            overlaps_sequence = not unsequenced_insert_locus.overlapsAnysense(breakpoint)
        else:
            overlaps_sequence = True

        overlaps[str(breakpoint)] = (cur_overlap, overlaps_sequence, extension)
        # best_overlap = max(cur_overlap, best_overlap)
        # if cur_overlap > best_overlap:
        #     best_breakpoint = breakpoint
        #     best_overlap = cur_overlap

    return overlaps#best_overlap, best_breakpoint

def set_read_supports_allele(aln_set, aln, allele, score, read_stats, breakpoint_collection, min_overlap):
    if not aln.concordant(read_stats):
        return 0

    assert len(aln.loci) == 1
    aln_locus = aln.loci[0]
    try:
        chrom = aln.aln1.chrom
        start = max(aln.aln1.reference_start, aln.aln2.reference_start)
        end = min(aln.aln1.reference_end, aln.aln2.reference_end)

        unsequenced_insert_locus = intervals.Locus(chrom, start, end, "+")
    except AttributeError:
        unsequenced_insert_locus = None

    try:
        if aln.insert_size > read_stats.max_reasonable_insert_size():
            return 0
        if aln.insert_size < read_stats.min_reasonable_insert_size():
            return 0
    except (IndexError, AttributeError):
        pass

    overlaps = get_overlaps(aln_locus, unsequenced_insert_locus, breakpoint_collection)
    if len(overlaps) == 0:
        return 0

    best_overlap = max(list(zip(*overlaps.values()))[1])

    aln_set.supports_allele = allele
    aln_set.support_prob = score / 40.0 #(1 - mapq.phred_to_prob(score, 10.0))
    aln_set.supporting_aln = aln

    aln.set_tag("OV", best_overlap)
    aln.set_tag("Ov", json.dumps(overlaps))
    aln.overlap = best_overlap

    return aln_set.support_prob

def get_best_aln(alns):
    if len(alns) == 0:
        return None
    consider_alns = [alns[0]]
    aln1_qual = alns[0].score
    for aln in alns[1:]:
        if debug.IS_DEBUG:
            print("aln.score", aln.score, aln.matched_count, aln.read_count_in_region,abs(aln.reference_end - aln.reference_start),abs(alns[0].reference_end - alns[0].reference_start),(aln.score > aln1_qual and abs((aln.reference_end - aln.reference_start) - aln.ctg_len) < 500),aln.q_en,aln._read.query_length,len(aln._read.query_sequence), aln._read.query_name)
        if abs(aln.score - aln1_qual) <= 200 or (aln.score > aln1_qual and
                                                 (abs((aln.reference_end - aln.reference_start) - aln.ctg_len) - abs((alns[0].reference_end - alns[0].reference_start) - alns[0].ctg_len) < 1000) or aln.original_seq_len - aln.q_en < 100):
            # print("aln.score", aln.score, aln.matched_count, aln.read_count_in_region, aln._read.query_name)
            consider_alns.append(aln)
    consider_alns.sort(key=lambda x: x.matched_count, reverse=True)
    return consider_alns[0]
def check_diff_region(aln, diff_len, percent, is_dup_ins=False):
    if aln is None:
        return False,0,0, percent
    original_percent = percent
    if diff_len < 500:
        percent = percent - 0.05
    if diff_len < 200:
        percent = percent - 0.05
    if diff_len > 10000:
        percent = percent - 0.05
    if percent == original_percent and is_dup_ins:
        percent = percent - 0.05
    if aln.read_count_in_region == 0 or aln.matched_count == 0:
        return False,0,0, percent
    percent1 = aln.matched_count / aln.read_count_in_region
    percent2 = min(aln.matched_count, diff_len) / max(aln.matched_count, diff_len)
    # the alignment covers the diff region but not cover full
    if debug.IS_DEBUG:
        print(aln.ctg_len ,aln._read.reference_start, aln.reference_end,diff_len > 10000,(aln.reference_end - (debug.ALIGN_DISTANCE - aln.reference_start))/diff_len > 0.25 , aln.read_count_in_region / diff_len,diff_len, "diff_len", aln._read.query_name)
    # that only cover the diff region less than 0.5
    if (diff_len < 2000 and aln.matched_count / diff_len < 0.5) or (diff_len > 2000 and diff_len < 10000 and aln.matched_count / diff_len < 0.3):
        # if aln.read_count_in_region / diff_len < 0.5:
        return False,0,0, percent
    # for alt, aln.ctg_len - aln.reference_end >  debug.ALIGN_DISTANCE
    if (diff_len > 10000 and (aln.reference_end - (debug.ALIGN_DISTANCE - aln.reference_start))/diff_len > 0.25) or ((aln.reference_end - debug.ALIGN_DISTANCE)/diff_len > 0.5 and aln.matched_count / diff_len > 0.5):
        percent1 = aln.matched_count / aln.read_count_in_region
        f1 = percent1 >= percent
        if debug.IS_DEBUG:
            print(f1, percent1, percent2, aln._read.query_name, percent, "percent1")
        if (aln.reference_end - (debug.ALIGN_DISTANCE - aln.reference_start))/diff_len > 1:
            f2 = percent2 >= percent
            if is_dup_ins:
                f2 = (percent2 >= percent - 0.05)
            print(f1, percent1, percent2, aln._read.query_name, percent, "percent2")
            return f1 and f2, min(percent1, percent2), max(percent1, percent2), percent
        return f1, percent1, percent1, percent

    else:
        f1 = percent1 >= percent
        f2 = percent2 >= percent
        if debug.IS_DEBUG:
            print(f1, percent1, percent2, aln._read.query_name, percent, "percent3")
    # if in_region:
    #     return f1 and f2
    # else:
        return f1 and f2, min(percent1, percent2), max(percent1, percent2), percent


def assign_reads_to_alleles(aln_sets, ref_breakpoint_collection, alt_breakpoint_collection, read_stats, diff_len, args):
    def get_best_score(_aln_set, _allele):
        if _allele == "ref":
            alignments = _aln_set.ref_pairs
        elif _allele == "alt":
            alignments = _aln_set.alt_pairs
        if len(alignments) > 0:
            return alignments[0].mapq
        return 0
    percent = args.percent
    ref_total = 0
    alt_total = 0

    if args.aligner == "minimap2":

        for aln_set in aln_sets:
            ref_score = get_best_score(aln_set, "ref")
            alt_score = get_best_score(aln_set, "alt")

            # if aln_set.name == "D00360:99:C8VWFANXX:4:2310:5190:27306":
            aln_set.supports_allele = "amb"
            aln_set.support_prob = 0
            aln_set.supporting_aln = None
            # print(aln_set._read)
            # print(">REF<")
            # print("ref_score", ref_score)
            # print("alt_score", alt_score)
            # for aln in aln_set.ref_pairs:
                # print(" ", aln.aln1.locus, aln.aln1.cigarstring, aln.aln1.score)
                # print(" ", aln.aln2.locus, aln.aln2.cigarstring, aln.aln2.score)
                # print(" ", aln.mapq)
            # print(">ALT<")
            # for aln in aln_set.alt_pairs:
                # print(" ", aln.aln1.locus, aln.aln1.cigarstring, aln.aln1.score)
                # print(" ", aln.aln2.locus, aln.aln2.cigarstring, aln.aln2.score)
                # print(" ", aln.mapq)
            # diff_len = abs(aln_set.ref_pairs[0].ctg_len - aln_set.alt_pairs[0].ctg_len)
            if debug.IS_DEBUG:
                for item in aln_set.ref_pairs:
                    print(item.matched_count, item.read_count_in_region, item.score, item._read.query_name, item.reference_end, "item ref")
                for item in aln_set.alt_pairs:
                    print(item.matched_count, item.read_count_in_region, item.score, item._read.query_name, item.reference_end, "item alt")
            ref_best_aln = get_best_aln(aln_set.ref_pairs)
            alt_best_aln = get_best_aln(aln_set.alt_pairs)
            # TODO, any other situation?
            is_dup_ins = False
            if ref_best_aln is not None and alt_best_aln is not None:
                is_dup_ins = alt_best_aln.ctg_len > ref_best_aln.ctg_len
            is_ref, ref_percent, ref_percent2, current_percent = check_diff_region(ref_best_aln, diff_len, percent, is_dup_ins)
            print("xxxxxxxxxxxxxxxxxxxxxx")
            is_alt, alt_percent, alt_percent2, current_percent = check_diff_region(alt_best_aln, diff_len, percent, is_dup_ins)
            # TODO if diff len large than 300, 0.95 and marge > 0.1, if diff len less than 300 and large than 150, 0.9 and marge 0.5, if diff len less than 150, 0.85 and marge 0.5
            ref_alt_margin = 0.3
            if math.isclose(percent - current_percent, 0.1, abs_tol=0.0001):
                ref_alt_margin = 0.3
            if is_dup_ins and diff_len < 200:
                ref_alt_margin = ref_alt_margin - 0.03
            large_than_margin = False
            large_than_margin = max(abs(ref_percent2 - alt_percent), abs(alt_percent2 - ref_percent)) > ref_alt_margin
            large_than_margin_and_05 = max(abs(ref_percent2 - alt_percent), abs(alt_percent2 - ref_percent)) > ref_alt_margin and math.isclose(ref_alt_margin,0.5, abs_tol=0.0001)
            if debug.IS_DEBUG:
                print(is_ref, is_alt,ref_percent, ref_percent2, alt_percent, alt_percent2, diff_len,ref_score,alt_score, alt_percent,abs(ref_percent - alt_percent), abs(ref_percent - alt_percent) < ref_alt_margin,aln_set.query_name,large_than_margin,ref_alt_margin,percent,current_percent, "is ref alt")
            aln_ref = None
            aln_alt = None
            if len(aln_set.ref_pairs) != 0:
                aln_ref = aln_set.ref_pairs[0]
            if len(aln_set.alt_pairs) != 0:
                aln_alt = aln_set.alt_pairs[0]
            if (is_ref and is_alt) or (not is_ref and not is_alt):
                aln = aln_ref if aln_ref != None else aln_alt
                aln_set.supports_allele = "amb"
                aln_set.supporting_aln = aln
                # print(aln_set.query_name, "amb")
            elif (is_ref and large_than_margin) or (is_ref and large_than_margin_and_05 and ref_percent > alt_percent):
                if alt_percent2 > 0.95 and alt_percent > 0.75:
                    aln = aln_ref
                    aln_set.supports_allele = "amb"
                    aln_set.supporting_aln = aln
                else:
                    aln = aln_ref
                    # if is_ref:
                    ref_total += set_read_supports_allele(
                        aln_set, aln, "ref", ref_score, read_stats, ref_breakpoint_collection, min_overlap=4)
                # print(aln_set.query_name, "ref")
            elif (is_alt and large_than_margin) or (is_alt and large_than_margin_and_05 and ref_percent < alt_percent):
            # elif (alt_score - ref_score > 1 and alt_score >= 30 and large_than_margin) or (large_than_margin_and_05 and ref_percent < alt_percent):
                if ref_percent2 > 0.95 and ref_percent > 0.75:
                    aln = aln_alt
                    aln_set.supports_allele = "amb"
                    aln_set.supporting_aln = aln
                else:
                    aln = aln_alt
                    alt_total += set_read_supports_allele(
                        aln_set, aln, "alt", alt_score, read_stats, alt_breakpoint_collection, min_overlap=4)
                # print(aln_set.query_name, "alt")
            else:
                aln = aln_ref if aln_ref != None else aln_alt
                aln_set.supports_allele = "amb"
                aln_set.supporting_aln = aln
            # print(aln_set.query_name, "final amb")
    else:
        for aln_set in aln_sets:
            ref_score = get_best_score(aln_set, "ref")
            alt_score = get_best_score(aln_set, "alt")

            # if aln_set.name == "D00360:99:C8VWFANXX:4:2310:5190:27306":
            aln_set.supports_allele = "amb"
            aln_set.support_prob = 0
            aln_set.supporting_aln = None

            if ref_score - alt_score > 1:
                # print(aln_set.name)
                # print(">REF<")
                # for aln in aln_set.ref_pairs:
                #     print(" ", aln.aln1.locus, aln.aln1.cigarstring, aln.aln1.score)
                #     print(" ", aln.aln2.locus, aln.aln2.cigarstring, aln.aln2.score)
                #     print(" ", aln.mapq)
                # print(">ALT<")
                # for aln in aln_set.alt_pairs:
                #     print(" ", aln.aln1.locus, aln.aln1.cigarstring, aln.aln1.score)
                #     print(" ", aln.aln2.locus, aln.aln2.cigarstring, aln.aln2.score)
                #     print(" ", aln.mapq)

                aln = aln_set.ref_pairs[0]
                ref_total += set_read_supports_allele(
                    aln_set, aln, "ref", ref_score, read_stats, ref_breakpoint_collection, min_overlap=4)


            elif alt_score - ref_score > 1:
                aln = aln_set.alt_pairs[0]
                alt_total += set_read_supports_allele(
                    aln_set, aln, "alt", alt_score, read_stats, alt_breakpoint_collection, min_overlap=4)
            elif len(aln_set.ref_pairs) > 0:
                aln = aln_set.ref_pairs[0]
                aln_set.supports_allele = "amb"
                aln_set.supporting_aln = aln
    return ref_total, alt_total


def test():
    print(calculate_genotype_likelihoods(0, 53))
    # print(calculate_genotype_likelihoods(13,2))
    # print(calculate_genotype_likelihoods(2,40))
    # print(calculate_genotype_likelihoods(25,26))


if __name__ == '__main__':
    test()
