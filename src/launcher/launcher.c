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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

const char *argp_program_version = "panoptes";

struct {
    const char *key;
    const char *name;
} tools[] = {{"memcheck", "MEMCHECK"}};
const size_t n_tools = sizeof(tools) / sizeof(tools[0]);

struct parse {
    /*
     * When libpanoptes.so is not in the library search path (particularly when
     * building Panoptes and running its tests), we want to support injection
     * of the library from the CMake test harness into the launcher.  This
     * option is hidden from users.
     */
    const char *library_path;
    struct stat library_stat;

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
        case 2:
            /* Library path. */
            if (arg) {
                /*
                 * Stat the argument to verify that it is a file or directory,
                 * caching the result.
                 */
                int ret = stat(arg, &p->library_stat);
                if (ret < 0) {
                    argp_error(state, "Unable to stat '%s'.", arg);
                } else if (!(p->library_stat.st_mode & (S_IFREG | S_IFDIR))) {
                    argp_error(state, "'%s' is not a file or directory.", arg);
                }

                p->library_path = arg;
            }

            break;
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
        {"tool",    1, "<name>", 0,             tool_doc, 0},
        {"library", 2, "<lib>",  OPTION_HIDDEN, 0,        0},
        {0,         0, 0,        0,             0,        0}};
    const char args_doc[] = "[options] program [program options]";

    struct argp argp = {options, parse_opt, args_doc, doc, 0, 0, 0};
    struct parse p;
    p.library_path = NULL;
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
     * Preload string.  We estimate the needed size loosely.
     */
    const char libpanoptes[] = "libpanoptes.so";
    char *preload_string;
    size_t preload_strlen = sizeof(libpanoptes) + sizeof(preload_env);
    if (p.library_path) {
        preload_strlen += strlen(p.library_path);
    }
    preload_string = malloc(preload_strlen);

    int ret = 0;
    if (!(preload_string)) {
        fprintf(stderr, "Unable to allocate memory.\n");
        ret = 1;
        goto end;
    }

    if (p.library_path && (p.library_stat.st_mode & S_IFDIR)) {
        /*
         * The null terminator from both libpanotes and preload environment
         * gives us an extra character for the /.
         */
        snprintf(preload_string, preload_strlen, "%s%s/%s", preload_env,
            p.library_path, libpanoptes);
    } else if (p.library_path && (p.library_stat.st_mode & S_IFREG)) {
        snprintf(preload_string, preload_strlen, "%s%s", preload_env,
            p.library_path);
    } else if (p.library_path) {
        fprintf(stderr, "Specified library path is not a file or directory.\n");
        ret = 1;
        goto free_preload;
    } else {
        snprintf(preload_string, preload_strlen, "%s%s", preload_env,
            libpanoptes);
    }

    /* Format a string for the tool. */
    char tool_string[256];
    snprintf(tool_string, sizeof(tool_string), "%s%s", panoptes_env,
        tools[p.tool].name);

    /* 3. Allocate and copy the list. */
    char **new_env = malloc(sizeof(*new_env) * i);
    if (!(new_env)) {
        fprintf(stderr, "Unable to allocate memory.  Aborting.\n");
        ret = 1;
        goto free_preload;
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

    ret = execvpe(file, argv + last, new_env);
    /* We shouldn't return, but if we do, clean up. */
    free(new_env);
    free_preload:
    free(preload_string);
    end:
    return ret;
}
