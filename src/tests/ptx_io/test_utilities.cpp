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

#include <cstdlib>
#include <errno.h>
#include <fcntl.h>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <tests/ptx_io/test_utilities.h>
#include <unistd.h>

using namespace panoptes;

ptx_checker::ptx_checker() { }
ptx_checker::~ptx_checker() { }

bool ptx_checker::check(const std::string & program) {
    return check(program, std::vector<std::string>());
}

bool ptx_checker::check(const std::string & program,
        const std::vector<std::string> & args) {
    int pipes_in[2];
    pipe(pipes_in);

    char file[] = "ptxas";

    char arg0[] = "ptxas";
    char arg1[] = "-o";
    char arg2[] = "/dev/null";
    char arg3[] = "-";

    std::vector<char *> argv;
    argv.push_back(arg0);
    argv.push_back(arg1);
    argv.push_back(arg2);
    argv.push_back(arg3);
    size_t n_args = args.size();
    for (size_t i = 0; i < n_args; i++) {
        /*
         * It's never fun to use const_cast. We do not expect execvp to mutate
         * its argv argument, so the hassle of allocating char* arrays is
         * wasteful.
         */
        argv.push_back(const_cast<char *>(args[i].c_str()));
    }
    argv.push_back(NULL);

    int pid = fork();
    if (pid == -1) {
        std::string error("Unable to fork.");
        throw std::runtime_error(error);
    } else if (pid == 0) {
        /* Child.  Setup pipe as input. */
        dup2(pipes_in[0], STDIN_FILENO);
        close(pipes_in[0]);
        close(pipes_in[1]);

        /* Setup /dev/null as stderr/stdout. */
        int dev_null = open("/dev/null", O_WRONLY);
        if (dev_null < 0) {
            exit(1);
        }

        dup2(dev_null, STDOUT_FILENO);
        dup2(dev_null, STDERR_FILENO);
        close(dev_null);

        execvp(file, argv.data());
        /* execvp returning implies a error occured. */
        exit(1);
    } else {
        /* Parent. */
        int in = pipes_in[1];
        close(pipes_in[0]);

        const size_t bytes = program.size();
        size_t offset = 0;
        while (offset < bytes) {
            ssize_t consumed =
                write(in, program.c_str() + offset, bytes - offset);
            if (consumed < 0) {
                if (errno == EINTR) {
                    continue;
                }

                return false;
            }

            offset += static_cast<size_t>(consumed);
        }

        close(in);

        int status;
        (void) waitpid(pid, &status, 0);

        bool ret;
        if (WIFEXITED(status)) {
            ret = WEXITSTATUS(status) == 0;
        } else {
            ret = false;
        }

        return ret;
    }
}
