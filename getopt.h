#ifndef CUDABSGS_GETOPT_H
#define CUDABSGS_GETOPT_H

#include <string.h>

static char *optarg = NULL;
static int optind = 1;
static int opterr = 1;
static int optopt = 0;

static int getopt(int argc, char *const argv[], const char *optstring) {
	static const char *next = NULL;
	const char *opt = NULL;

	(void)opterr;
	optarg = NULL;

	if(next == NULL || *next == '\0') {
		if(optind >= argc || argv[optind] == NULL || argv[optind][0] != '-' || argv[optind][1] == '\0') {
			return -1;
		}
		if(strcmp(argv[optind], "--") == 0) {
			optind++;
			return -1;
		}
		next = argv[optind] + 1;
		optind++;
	}

	optopt = (unsigned char)*next++;
	opt = strchr(optstring, optopt);
	if(opt == NULL || optopt == ':') {
		return '?';
	}

	if(opt[1] == ':') {
		if(*next != '\0') {
			optarg = (char *)next;
			next = NULL;
		}
		else if(optind < argc) {
			optarg = argv[optind++];
			next = NULL;
		}
		else {
			return optstring[0] == ':' ? ':' : '?';
		}
	}

	return optopt;
}

#endif
