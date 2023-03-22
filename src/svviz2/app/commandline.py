import argparse
import sys

import svviz2

def visualization_file_format(original_string):
    string = original_string.lower()
    if string not in ["pdf", "png", "svg"]:
        msg = "Invalid output file format: '{}'".format(string)
        raise argparse.ArgumentTypeError(msg)
    return string

def parse_args(input_args):
    parser = argparse.ArgumentParser(description="svviz2 version {}".format(svviz2.__version__),
        usage="%(prog)s [options] --ref REF --variants VARIANTS BAM [BAM2 ...]",
        formatter_class=argparse.RawTextHelpFormatter)

    required_args = parser.add_argument_group("Required arguments")

    required_args.add_argument("bam", nargs="+", help=
        "sorted, indexed bam file containing reads of interest to plot; can be \n"
        "specified multiple times to load multiple samples")

    required_args.add_argument("--ref", "-r", help=
        "reference fasta file (a .faidx index file will be created if it doesn't \n"
        "exist so you need write permissions for this directory)")

    required_args.add_argument("--variants", "-V", help=
        "the variants to analyze, in vcf or bcf format (vcf files may be \n"
        "compressed with gzip)")


    optional_args = parser.add_argument_group("Optional arguments")

    optional_args.add_argument("--prepar", action="store_true", help=
        "if prepar")
    optional_args.add_argument("--sub_fa", type=str, help=
        "ref and alt fa")

    optional_args.add_argument("--outdir", "-o", type=str, help=
        "output directory for visualizations, summaries, etc (default: current \n"
        "working directory)")
    optional_args.add_argument("--support_file", type=str, help=
        "output directory for visualizations, summaries, etc (default: current \n"
        "working directory)")

    optional_args.add_argument("--format", type=visualization_file_format, default="pdf", help=
        "format for output visualizations; must be one of pdf, png or svg \n"
        "(default: pdf, or svg if no suitable converter is found)")

    optional_args.add_argument("--savereads", action="store_true", help=
        "output the read realignments against the appropriate alt or ref allele \n"
        "(default: false)")

    optional_args.add_argument("--min-mapq", type=int, help=
        "only reads with mapq>=MIN_MAPQ will be analyzed; when analyzing \n"
        "paired-end data, at least one read end must be near the breakpoints \n"
        "with this mapq (default:0)")
    
    optional_args.add_argument("--align-distance", type=int, help=
        "sequence upstream and downstream of breakpoints to include when \n"
        "performing re-alignment (default: infer from data)")
    
    optional_args.add_argument("--batch-size", type=int, default=10000, help=
        "Number of reads to analyze at once; larger batch-size values may run \n"
        "more quickly but will require more memory (default=10000)")

    optional_args.add_argument("--downsample", type=int, help=
        "Ensure the total number of reads per event per sample does not exceed \n"
        "this number by downsampling (default: infinity)")

    optional_args.add_argument("--aligner", type=str, default="bwa", help=
        "The aligner to use for realigning reads; either ssw (smith-waterman) or \n"
        "bwa (default=bwa)")

    optional_args.add_argument("--only-realign-locally", action="store_true", help=
        "Only when using bwa as the aligner backend, when this option is enabled,\n"
        "reads will only be aligned locally around the breakpoints and not also \n"
        "against the full reference genome (default: False)")
    
    optional_args.add_argument("--fast", action="store_true", help=
        "More aggressively skip reads that are unlikely to overlap\n"
        "the breakpoints (default: false)")
    
    optional_args.add_argument("--first-variant", type=int, help=
        "Skip all variants before this variant; counting starts with first \n"
        "variant in input VCF as 0 (default: 0)")

    optional_args.add_argument("--last-variant", type=int, help=
        "Skip all variants after this variant; counting starts with first \n"
        "variant in input VCF as 0 (default: end of vcf)")

    optional_args.add_argument("--render-only", action="store_true", help="")
    optional_args.add_argument("--no-render", action="store_true", help="")
    optional_args.add_argument("--dotplots-only", action="store_true", help="")
    optional_args.add_argument("--no-dotplots", action="store_true", help="")
    optional_args.add_argument("--report-only", action="store_true", help="")
    optional_args.add_argument("--no-report", action="store_true", help="")

    optional_args.add_argument("--only-plot-context", type=int, help=
        "Only show this many nucleotides before the first breakpoint, and the \n"
        "last breakpoint in each region (default: show as much context as needed \n"
        "to show all reads fully)")

    optional_args.add_argument("--also-plot-context", type=int, help=
        "Generates two plots per event, one using the default settings, and one \n"
        "generatedby zooming in on the breakpoints as per the \n"
        "--only-plot-context option")

    if len(input_args)<1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(input_args)
    if args.only_plot_context and args.also_plot_context:
        parser.print_help()
        print("ERROR: you only want to use one of --only-plot-context or --also-plot-context")
        sys.exit(1)

    return args
