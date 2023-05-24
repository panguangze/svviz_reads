import collections
from pysam.libcalignedsegment cimport AlignedSegment
import numpy
cimport numpy

ctypedef numpy.int_t DTYPE_INT_t
ctypedef numpy.float_t DTYPE_FLOAT_t
cimport cython
from libc.stdint cimport uint32_t
import svviz2.debug as debug

cdef:
    float match = 0.0
    float mismatch= 0.0
    float gap_open= -1.0
    float gap_extend= -1.0
    float clipping_penalty= 0.0
    float min_clip_length= 10
    float discordant_penalty= -15
    float phred_scale= 10.0
    float min_base_quality= 0
    float max_base_quality= 10
cdef:
    char *TAG_END_SCORE = "Es"
    float MIN_PHRED_QUALITY = 3.0
    int BAM_CHARD_CLIP = 5




cdef phred_to_prob(float phred, floatphred_scale):
    if phred < 0:
        phred = 0
    return 10.0 ** (phred / -phred_scale)

cdef prob_to_phred(float phred, float scale):
    if phred <= 0:
        return numpy.inf

    return -scale * numpy.log10(phred)


cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=1] get_qualities(aln, float max_quality):
    if aln.query_qualities is None:
        qualities = numpy.repeat(40, aln.query_length)
    else:
        qualities = numpy.asarray(aln.query_qualities, dtype=float)

    qualities[qualities < MIN_PHRED_QUALITY] = MIN_PHRED_QUALITY

    scale = max_base_quality - min_base_quality
    qualities = (qualities / max_quality) * scale + min_base_quality

    return qualities



cpdef double get_alignment_end_score(
    AlignedSegment aln, str ref_seq, bint add_tag=True, float max_quality=40.0) except 9999: 

    cdef:
        numpy.ndarray[DTYPE_FLOAT_t, ndim=1] qualities
        float clip_left_adjust, clip_right_adjust
        double log10_score = 0
        float read_quality
        uint32_t query_start, query_end, reference_start
        char* query_sequence
        char* reference_sequence
        char ref_nuc, read_nuc

        bint in_gap = False
        bint read_has_nuc, ref_has_nuc

        int i = 0

    qualities = get_qualities(aln, max_quality)
    cigartuples = aln.cigartuples
    clip_left_adjust =  min(1.0, cigartuples[0][1] / min_clip_length)
    clip_right_adjust = min(1.0, cigartuples[-1][1] / min_clip_length)

    reference_start = aln.reference_start

    _query_sequence = aln.query_sequence.encode("ascii")
    query_sequence = _query_sequence
    query_start = aln.query_alignment_start
    query_end = aln.query_alignment_end

    _reference_sequence = ref_seq.encode("ascii")
    reference_sequence = _reference_sequence


    tag = False
    for read_pos, ref_pos in aln.get_aligned_pairs():
        # if ref_pos is not None and ref_pos < 5000:
        #     tag = True
        #     continue
        # if ref_pos is None and not tag:
        #     continue
        read_has_nuc = False
        if read_pos is not None:
            read_nuc = query_sequence[read_pos]
            read_quality = qualities[read_pos]
            read_has_nuc = True
        ref_has_nuc = False
        if ref_pos is not None:
            ref_nuc = reference_sequence[ref_pos-reference_start]
            ref_has_nuc = True
        # print(i, read_pos, read_nuc, ref_pos, ref_nuc, log10_score, read_has_nuc, ref_has_nuc)

        if i < query_start:
            log10_score += (clip_left_adjust) * (read_quality / -phred_scale + clipping_penalty)
        elif i >= query_end:
            log10_score += (clip_right_adjust) * (read_quality / -phred_scale + clipping_penalty)
        elif not read_has_nuc:
            # deletion
            if in_gap:
                log10_score += read_quality / -phred_scale + gap_extend
                # print(read_quality / -phred_scale + gap_extend, "delete gap_extend")
            else:
                log10_score += read_quality / -phred_scale + gap_open
                # print(read_quality / -phred_scale + gap_extend, "delete gap open")
                in_gap = True
        elif not ref_has_nuc:
            # insertion
            if in_gap:
                log10_score += read_quality / -phred_scale + gap_extend
            else:
                log10_score += read_quality / -phred_scale + gap_open
                in_gap = True
        elif read_nuc != ref_nuc:
            in_gap = False
            log10_score += read_quality / -phred_scale + mismatch
        else:
            in_gap = False
            prob = 1 - phred_to_prob(read_quality, phred_scale)
            log10_score += prob_to_phred(prob, -1) + match

        i += 1

    #     # print(read_pos, read_seq, ref_pos, ref_seq, match)
    # if aln.cigartuples[0][0] == BAM_CHARD_CLIP:
    #     raise Exception("hard clipping not yet supported") # should match soft-clipping
    #     # log10_score += read_quality / -phred_scale + gap_open"]
    #     log10_score += (read_quality / -phred_scale + clipping_penalty) * (aln.cigartuples[0][1])
    # if aln.cigartuples[-1][0] == BAM_CHARD_CLIP:
    #     raise Exception("hard clipping not yet supported") # should match soft-clipping
    #     # log10_score += read_quality / -phred_scale + gap_open"]
    #     log10_score += (read_quality / -phred_scale + clipping_penalty) * (aln.cigartuples[-1][1])

    return log10_score



cpdef tuple diff_region_similarity(
    AlignedSegment aln, int q_st, int q_en, str ref_seq, int diff_len, bint add_tag=True, float max_quality=40.0):

    cdef:
        numpy.ndarray[DTYPE_FLOAT_t, ndim=1] qualities
        float clip_left_adjust, clip_right_adjust
        double log10_score = 0
        float read_quality
        uint32_t query_start, query_end, reference_start
        char* query_sequence
        char* reference_sequence
        char ref_nuc, read_nuc

        bint in_gap = False
        bint read_has_nuc, ref_has_nuc

        int i = 0


    qualities = get_qualities(aln, max_quality)
    cigartuples = aln.cigartuples
    clip_left_adjust =  min(1.0, cigartuples[0][1] / min_clip_length)
    clip_right_adjust = min(1.0, cigartuples[-1][1] / min_clip_length)

    reference_start = aln.reference_start

    query_start = q_st
    query_end = q_en
    i = query_start

    # print(aln.query_sequence, "query_sequence")
    # print(ref_seq, "ref_sequence")
    # print(query_start, query_end, "start end")

    _query_sequence = aln.query_sequence.encode("ascii")
    query_sequence = _query_sequence

    _reference_sequence = ref_seq.encode("ascii")
    reference_sequence = _reference_sequence

    match_count = 0
    unmatch_count = 0
    read_base_count = 0
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0

    tag = True
    for read_pos, ref_pos in aln.get_aligned_pairs():
        # print(read_pos, ref_pos, "read ref")
        if ref_pos is not None and ref_pos < debug.ALIGN_DISTANCE:
            tag = True
            continue
        if ref_pos is None and tag:
            continue
        if ref_pos is not None and ref_pos > debug.ALIGN_DISTANCE+ diff_len:
            break
        tag = False
        read_has_nuc = False
        if read_pos is not None:
            read_nuc = query_sequence[read_pos]
            read_quality = qualities[read_pos]
            read_has_nuc = True
            read_base_count += 1
        ref_has_nuc = False
        if ref_pos is not None:
            ref_nuc = reference_sequence[ref_pos]
            ref_has_nuc = True
        # print(i, read_pos, read_nuc, ref_pos, ref_nuc, log10_score, read_has_nuc, ref_has_nuc)
        # print(read_nuc, ref_nuc, "read_nuc, ref_nuc")
        if i < query_start:
            # i1 += 1
            pass
        elif i >= query_end:
            # i2 += 1
            pass
        elif not read_has_nuc:
            i3 += 1
            # pass
            # print("read_has_nuc", read_has_nuc)
        elif not ref_has_nuc:
            i4 += 1
            # pass
            # print("ref_has_nuc", ref_has_nuc)
        elif read_nuc != ref_nuc:
            # pass
            # match_count += 1
            unmatch_count += 1
        else:
            match_count += 1
        i += 1
    # print(i,"index")

    # print(match_count,unmatch_count,i3,i4, read_base_count, match_count/diff_len,aln.query_name, query_start, query_end, reference_start, "match_count read_base_count")

    return match_count, read_base_count
