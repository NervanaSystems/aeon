#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <syslog.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <string>

#include "status.h"
#include "pidfile.h"
#include "utils.h"
#include "server.hpp"

/* Global timestamp value. It shell be used to update a timestamp field of block
   device structure. See block.h for details. */
time_t timestamp = 0;

/**
 * @brief Daemon process termination flag.
 *
 * This flag indicates that daemon process should terminate. User must send
 * SIGTERM to daemon in order to terminate the process gently.
 */
static sig_atomic_t terminate = 0;

/**
 * @brief Aeon service finalize function.
 *
 * This is internal function of aeon service. It is used to finalize daemon
 * process i.e. free allocated memory, unlock and remove pidfile and close log
 * file and syslog. The function is registered as on_exit() handler.
 *
 * @param[in]     status          The function ignores this parameter.
 * @param[in]     progname        The name of the binary file. This argument
 *                                is passed via on_exit() function.
 *
 * @return The function does not return a value.
 */
static void _aeonsvc_fini(int __attribute__((unused)) status, void* progname)
{
    log_close();
    pidfile_remove(static_cast<char*>(progname));
}

/**
 * @brief Puts exit status to a log file.
 *
 * This is internal function of aeon service. It is used to report an exit
 * status of the aeon service. The message is logged in to syslog and to log
 * file. The function is registered as on_exit() hander.
 *
 * @param[in]     status            Status given in the last call to exit()
 *                                  function.
 * @param[in]     ignore            Pointer to placeholder where ignore flag is
 *                                  stored. If flag is set 0 then parent process
 *                                  is exiting, otherwise a child is exiting.
 *                                  This argument must not be NULL pointer.
 *
 * @return The function does not return a value.
 */
static void _aeonsvc_status(int status, void* ignore)
{
    if (*((int*)ignore) != 0)
        log_info("exit status is %s.", strstatus(status));
    else if (status != STATUS_SUCCESS)
        log_error("parent exit status is %s.", strstatus(status));
}

/**
 * @brief SIGTERM handler function.
 *
 * This is internal function of aeon service.
 *
 * @param[in]    signum          - the number of signal received.
 *
 * @return The function does not return a value.
 */
static void _aeonsvc_sig_term(int signum)
{
    if (signum == SIGTERM)
    {
        log_info("SIGTERM caught - terminating daemon process.");
        terminate = 1;
    }
}

/**
 * @brief Configures signal handlers.
 *
 * This is internal function of aeon services. It sets to ignore SIGALRM,
 * SIGHUP and SIGPIPE signals. The function installs a handler for SIGTERM
 * signal. User must send SIGTERM to daemon process in order to shutdown the
 * daemon gently.
 *
 * @return The function does not return a value.
 */
static void _aeonsvc_setup_signals(void)
{
    struct sigaction act;
    sigset_t         sigset;

    sigemptyset(&sigset);
    sigaddset(&sigset, SIGALRM);
    sigaddset(&sigset, SIGHUP);
    sigaddset(&sigset, SIGTERM);
    sigaddset(&sigset, SIGPIPE);
    sigaddset(&sigset, SIGUSR1);
    sigprocmask(SIG_BLOCK, &sigset, NULL);

    act.sa_handler = SIG_IGN;
    act.sa_flags   = 0;
    sigemptyset(&act.sa_mask);
    sigaction(SIGALRM, &act, NULL);
    sigaction(SIGHUP, &act, NULL);
    sigaction(SIGPIPE, &act, NULL);
    act.sa_handler = _aeonsvc_sig_term;
    sigaction(SIGTERM, &act, NULL);
    sigaction(SIGUSR1, &act, NULL);

    sigprocmask(SIG_UNBLOCK, &sigset, NULL);
}

int main(int argc, char* argv[])
{
    int         daemon_flag = 0;
    std::string http_addr;
    std::string port;

    while (1)
    {
        static struct option long_options[] = {{"daemon", no_argument, &daemon_flag, 1},
                                               {"help", no_argument, 0, 'h'},
                                               {"address", required_argument, 0, 'a'},
                                               {"port", required_argument, 0, 'p'},
                                               {0, 0, 0, 0}};
        int option_index = 0;
        int c            = getopt_long_only(argc, argv, "h", long_options, &option_index);

        if (c == -1)
            break;
        switch (c)
        {
        case 'a': http_addr = optarg; break;
        case 'p': port      = optarg; break;
        case 'h':
            printf("Usage: %s ", argv[0]);
            printf("[--daemon] --address address --port port\n");
            return 0;
        case '?': return -1;
        }
    }
    if (http_addr.empty())
    {
        printf("Missing \"http_addr\" argument\n");
        return -1;
    }
    if (port.empty())
    {
        printf("Missing \"port\" argument\n");
        return -1;
    }

    http_addr.append(":");
    http_addr.append(port);

    if (!daemon_flag)
    {
        nervana::aeon_server server(http_addr);
        while (true)
            getchar();
    }
    else
    {
        verbose = VERB_ALL;
        printf("%s\n", "launching aeon service...");
        int i;

        set_invocation_name(argv[0]);
        printf("%s\n", "opening log...");
        openlog(progname, LOG_PID | LOG_PERROR, LOG_DAEMON);
        printf("%s\n", "opened.");

        if (getuid() != 0)
        {
            printf("%s\n", "need to be root.");
            log_error("Only root can run this application.");
            return STATUS_NOT_A_PRIVILEGED_USER;
        }

        if (on_exit(_aeonsvc_status, &terminate))
        {
            return STATUS_ONEXIT_ERROR;
        }

        if (pidfile_check(progname, NULL) == 0)
        {
            printf("%s\n", "process already running.");
            log_warning("daemon is running...");
            return STATUS_LEDMON_RUNNING;
        }

        pid_t pid = fork();
        if (pid < 0)
        {
            log_debug("main(): fork() failed (errno=%d).", errno);
            exit(EXIT_FAILURE);
        }
        if (pid > 0)
        {
            exit(EXIT_SUCCESS);
        }

        pid_t sid = setsid();
        if (sid < 0)
        {
            log_debug("main(): setsid() failed (errno=%d).", errno);
            exit(EXIT_FAILURE);
        }
        for (i = getdtablesize() - 1; i >= 0; --i)
            close(i);
        int t = open("/dev/null", O_RDWR);
        dup(t);
        dup(t);
        umask(027);

        if (chdir("/") < 0)
        {
            log_debug("main(): chdir() failed (errno=%d).", errno);
            exit(EXIT_FAILURE);
        }
        if (pidfile_create(progname))
        {
            log_debug("main(): pidfile_creat() failed.");
            exit(EXIT_FAILURE);
        }
        _aeonsvc_setup_signals();

        if (on_exit(_aeonsvc_fini, progname))
            exit(STATUS_ONEXIT_ERROR);

        log_info("aeon service has been started...");
        nervana::aeon_server server(http_addr);

        while (terminate == 0)
        {
            timestamp = time(NULL);
            log_debug("time %l", timestamp);
            sleep(5);
        }
        exit(EXIT_SUCCESS);
    }
}
