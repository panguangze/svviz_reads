import logging
import numpy

from svviz2.io import pairedreaditer
from svviz2.remap import alignment
from svviz2.remap.readpair import ReadPair
from svviz2.utility import misc

logger = logging.getLogger(__name__)

def get_read_batch(sample, datahub):
    if datahub.args.downsample:
        datahub.args.batch_size = numpy.inf
        batch = list(_get_read_batch(sample, datahub))[0]
        if len(batch) > datahub.args.downsample:
            numpy.random.seed(10)
            batch = list(numpy.random.choice(batch, size=datahub.args.downsample, replace=False))
        yield batch
    else:
        yield from _get_read_batch(sample, datahub)
        
def _get_read_batch(sample, datahub):
    read_filter = lambda x: x
    if sample.read_filter:
        read_filter = sample.read_filter
        
    if sample.single_ended:
        for batch in get_reads_unpaired(sample, datahub):
            yield read_filter(batch)
    else:
        for batch in get_read_pairs(sample, datahub):
            yield read_filter([ReadPair(read1, read2, sample.read_statistics) for (read1, read2) in batch])


def get_reads_unpaired(sample, datahub):
    logger.info("Loading more reads...")
    cur_reads = []
    search_regions = datahub.variant.search_regions(sample.search_distance)

    for region in search_regions:
        chrom, start, end = region.chrom, region.start, region.end
        chrom = misc.match_chrom_format(chrom, sample.bam.references)
        for read in sample.bam.fetch(chrom, start, end):
            # if read.query_name != "m150105_192231_42177R_c100761782550000001823161607221526_s1_p0/138972/39862_46995":
            #     continue

            if read.is_supplementary or read.is_duplicate or read.is_secondary:
                continue
            
            if datahub.args.min_mapq and read.mapq < datahub.args.min_mapq:
                continue

            cur_reads.append(alignment.Alignment(read))
            if datahub.args.batch_size is not None and len(cur_reads) >= datahub.args.batch_size:
                yield cur_reads
                logger.info("Loading more reads...")
                cur_reads = []

    yield cur_reads



def get_read_pairs(sample, datahub):
    """
    get batches of read-pairs -- this allows us to exhaustively search for mates without
    keeping everyone in memory
    """
    logger.info("Loading more read pairs...")
    cur_read_pairs = []
    search_distance = sample.search_distance
    search_regions = datahub.variant.search_regions(search_distance)
    paired_read_iter = pairedreaditer.PairedReadIter(sample.bam, search_regions)
    if datahub.args.min_mapq:
        paired_read_iter.pair_min_mapq = datahub.args.min_mapq

    import time
    t0 = time.time()
    
    for read_pair in paired_read_iter:
        # if read_pair[0].query_name != "HA2WPADXX:44:1:714777:0":
            # continue
        cur_read_pairs.append(read_pair)
        if datahub.args.batch_size is not None and len(cur_read_pairs) >= datahub.args.batch_size:
            t1 = time.time()
            logger.info("TIME to read batch: {:.1f}s".format(t1-t0))
            t0 = time.time()
            
            yield cur_read_pairs
            logger.info("Loading more read pairs...")
            cur_read_pairs = []
    t1 = time.time()
    logger.info("TIME to read batch: {:.1f}s".format(t1-t0))

    yield cur_read_pairs

    print("Reads with only N:", paired_read_iter.N_count)
