/**
 * Panoptes - A Binary Translation Framework for CUDA
 * (c) 2011-2013 Chris Kennelly <chris@ckennelly.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define _GNU_SOURCE /* For execvpe */

#include <argp.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *argp_program_version = "panoptes";

struct {
    const char *key;
    const char *name;
} tools[] = {{"memcheck", "MEMCHECK"}};
const size_t n_tools = sizeof(tools) / sizeof(tools[0]);

struct parse {
    size_t tool;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct parse *p = state->input;

    switch (key) {
        case 1:
            /*
             * We must check that arg is not NULL as passing --tool
             * (with no =...) leads to argp giving us a null pointer for arg.
             */
            if (arg) {
                size_t i;
                for (i = 0; i < n_tools; i++) {
                    if (strcmp(arg, tools[i].key) == 0) {
                        p->tool = i;
                        return 0;
                    }
                }

                argp_error(state, "Unknown tool '%s', try 'memcheck'.", arg);
            }

            argp_error(state, "No tool specified.");
        default:
            return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

int main(int argc, char **argv) {
    const char doc[] =
        "Panoptes: A Binary Translator for CUDA\n"
        "(c) 2011-2013 Chris Kennelly\n";

    const char tool_doc[] = "Use <name> translator [default: memcheck].";
    const struct argp_option options[] = {
        {"tool", 1, "<name>", 0, tool_doc, 0},
        {0,      0, 0,        0, 0,        0}};
    const char args_doc[] = "[options] program [program options]";

    struct argp argp = {options, parse_opt, args_doc, doc, 0, 0, 0};
    struct parse p;
    p.tool = 0;

    int last = 0;
    argp_parse(&argp, argc, argv, 0, &last, &p);

    if (last == argc) {
        /* We didn't get an executable.  Reject this with usage information. */
        argp_help(&argp, stderr, ARGP_HELP_STD_HELP, argv[0]);
        return 0;
    }

    const char *file = argv[last];
    /* Setup environment clone.
     *
     * 1. Count number of environment variables needed.
     */
    const char preload_env [] = "LD_PRELOAD=";
    const char panoptes_env[] = "PANOPTES_TOOL=";
    const size_t sentinel = (size_t) -1;
    size_t tool_index = sentinel, preload_index = sentinel;
    size_t i;
    for (i = 0; environ[i]; i++) {
        if (strncmp(environ[i], preload_env, sizeof(preload_env) - 1u) == 0) {
            /* We'll be overriding LD_PRELOAD. */
            preload_index = i;
        } else if (strncmp(environ[i], panoptes_env,
                sizeof(panoptes_env) - 1u) == 0) {
            /* The tool is set in the environment, so we'll be overriding it. */
            tool_index = i;
        }
    }

    if (preload_index == sentinel) {
        /* We're adding another environment variable. */
        i++;
    }

    if (tool_index == sentinel) {
        /* We're adding another environment variable. */
        i++;
    }

    /* We need a sentinel null terminator as well. */
    i++;

    /*
     * Preload string.  TODO:  In the future, if we expect libpanoptes.so to be
     * outside of the LD_LIBRARY_PATH, we may want to fill in a full path here.
     */
    char preload_string[] = "LD_PRELOAD=libpanoptes.so";

    /* Format a string for the tool. */
    char tool_string[256];
    snprintf(tool_string, sizeof(tool_string), "%s%s", panoptes_env,
        tools[p.tool].name);

    /* 3. Allocate and copy the list. */
    char **new_env = malloc(sizeof(*new_env) * i);
    if (!(new_env)) {
        fprintf(stderr, "Unable to allocate memory.  Aborting.\n");
        return 1;
    }

    for (i = 0; environ[i]; i++) {
        if (i == preload_index) {
            new_env[i] = preload_string;
        } else if (i == tool_index) {
            new_env[i] = tool_string;
        } else {
            new_env[i] = environ[i];
        }
    }

    /* 4. Add strings to the list if we didn't during processing. */
    if (preload_index == sentinel) {
        new_env[i] = preload_string;
        i++;
    }

    if (tool_index == sentinel) {
        new_env[i] = tool_string;
        i++;
    }

    /* 5. Null terminate. */
    new_env[i] = NULL;

    /* We shouldn't return, but if we do, clean up. */
    int ret = execvpe(file, argv + last, new_env);
    free(new_env);
    return ret;
}
