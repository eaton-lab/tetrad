#!/usr/bin/env python

"Parallelization with ipyparallel following ipyrad framework"

from __future__ import print_function
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import os
import sys
import time
import socket
import traceback
import subprocess
from random import getrandbits

import ipyparallel as ipp
from .utils import TetradError


class Parallel(object):
    """
    Connect or launch ipcluster and wrap jobs running on Client engines so 
    that engines can be interrupted or killed with a pleasant cleanup.
    """
    def __init__(self, tool, rkwargs=None, ipyclient=None, show_cluster=True, auto=False):

        # if no kwargs then empty dict
        if rkwargs is None:
            rkwargs = {}

        # the tool with a ._run() func and its run kwargs to be parallelized
        self.tool = tool
        self.rkwargs = rkwargs

        # parallel client connect or launch params
        self.ipyclient = ipyclient
        self.show_cluster = show_cluster
        self.auto = auto
        self.spacer = ""

        # require quiet attribute
        if not hasattr(self.tool, "quiet"):
            self.tool.quiet = False       


    def update_message(self, value):
        if not self.tool.quiet:
            print(value)


    def start_ipcluster(self):
        """
        The name is a unique id that keeps this __init__ of ipyrad distinct
        from interfering with other ipcontrollers. Run statements are wrapped
        so that ipcluster SHOULD be killed on exit.
        """
        # use random num for to cluster_id
        rand = getrandbits(32)
        self.tool.ipcluster["cluster_id"] = "ipp-{}".format(rand)

        # if engines=="MPI" then add --ip arg to view all sockets  
        iparg = ("--ip=*" if "MPI" in self.tool.ipcluster["engines"] else "")

        # make ipcluster arg call
        standard = [
            "ipcluster", "start",
            "--daemonize", 
            "--cluster-id={}".format(self.tool.ipcluster["cluster_id"]),
            "--engines={}".format(self.tool.ipcluster["engines"]),
            "--profile={}".format(self.tool.ipcluster["profile"]),
            "--n={}".format(self.tool.ipcluster["cores"]),
            "{}".format(iparg),
        ]
                   
        # wrap ipcluster start
        try:
            subprocess.check_call(
                standard,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE)

        # if cluster with THIS ID is running then kill it and try again
        except subprocess.CalledProcessError:
            subprocess.check_call([
                "ipcluster", "stop", 
                "--cluster-id", self.tool.ipcluster["cluster_id"],
            ], 
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
            )

            # after waiting try again to start it
            time.sleep(3)
            try:
                subprocess.check_call(
                    standard, 
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE)

            # if fails again then report it
            except subprocess.CalledProcessError as inst:
                print(inst)
                raise

        except Exception as inst:
            sys.exit("Error launching ipcluster for parallelization:\n({})\n"
                .format(inst))



    def wait_for_connection(self):
        """ 
        Creates a client to view ipcluster engines for a given profile and 
        returns it with at least one engine spun up and ready to go. If no 
        engines are found after nwait amount of time then an error is raised.
        If engines==MPI it waits a bit longer to find engines. If the number
        of engines is set then it waits even longer to try to find that number
        of engines.
        """
        # save stds for later, hide here to prevent ipp enforced print()
        save_stdout = sys.stdout 
        save_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # wrapped search for ipcluster
        try: 
            args = {
                "profile": self.tool.ipcluster["profile"],
                "timeout": self.tool.ipcluster["timeout"],
                "cluster_id": self.tool.ipcluster["cluster_id"],            
            }
            ipyclient = ipp.Client(**args)

            # restore std printing now that Client print statement has passed
            # sys.stdout = save_stdout
            # sys.stderr = save_stderr

            # allow time to find the connection; count cores to break
            for _ in range(6000):

                # how many cores can we find right now?
                ncores = len(ipyclient)
                self.update_message(
                    "Establishing parallel connection: {} cores"
                    .format(ncores))
                time.sleep(0.01)

                # If we know ncores, then wait for all 
                print(self.tool.ipcluster["cores"])
                if self.tool.ipcluster["cores"]:
                    time.sleep(0.1)
                    if ncores == self.tool.ipcluster["cores"]:
                        break

                # Looking for all available cores, auto stop 
                else:
                    
                    # If MPI and not all found break if no more in 3 secs
                    if self.tool.ipcluster["engines"] == "MPI":
                        # are any cores found yet? do long wait.
                        if ncores:
                            time.sleep(3)
                            if len(ipyclient) == ncores:
                                break

                    # if Local then wait 1 second between checks
                    else:
                        if ncores:
                            time.sleep(1.)
                            if len(ipyclient) == ncores:                            
                                break

        except KeyboardInterrupt as inst:
            raise inst

        except (IOError, OSError, ipp.TimeoutError, ipp.NoEnginesRegistered):
            raise TetradError(
                "\nipcluster not found, use 'auto=True' or see docs.")

        finally:
            # no matter what we reset the stds
            sys.stdout = save_stdout
            sys.stderr = save_stderr

        return ipyclient



    def get_cluster_info(self):
        """ reports host and engine info for an ipyclient """    
        # get engine data, skips busy engines.
        hosts = []
        for eid in self.ipyclient.ids:
            engine = self.ipyclient[eid]
            if not engine.outstanding:
                hosts.append(engine.apply(socket.gethostname))

        ## report it
        hosts = [i.get() for i in hosts]
        hostdict = {}
        for hostname in set(hosts):
            hostdict[hostname] = hosts.count(hostname)
        hpairs = [
            "{}: {} cores".format(key, val) for 
            (key, val) in hostdict.items()
        ]
        self.update_message(
            "{}Parallel connection | {}".format(
                self.spacer, " | ".join(hpairs)))



    def store_pids_for_shutdown(self):
        "reset tool ipcluster dict pids dict and set with current engine pids"
        self.tool.ipcluster["pids"] = {}
        for eid in self.ipyclient.ids:
            engine = self.ipyclient[eid]
            if not engine.outstanding:
                pid = engine.apply(os.getpid).get()
                self.tool.ipcluster["pids"][eid] = pid



    def wrap_run(self, dry_run=False):
        """
        Takes an analysis tools object with an associated _ipcluster attribute
        dictionary and either launches an ipcluster instance or connects to a 
        running one. The ipyclient arg overrides the auto arg.
        """

        # save a traceback of the stack on the remote engine that dies
        iptrace = None

        # wrap so we can shutdown ipp and format error and traceback
        try:
            # check that ipyclient is running by connecting (3 seconds tries)
            if self.ipyclient:
                for i in range(3):
                    if len(self.ipyclient):
                        break
                    else:
                        time.sleep(1)
                assert len(self.ipyclient), "ipcluster not connected/running."

            # launch ipcluster and get the parallel client with ipp-{} id
            elif self.auto:
                # set default to 4
                if not self.tool.ipcluster["cores"]:
                    self.tool.ipcluster["cores"] = detect_cpus()

                # start ipcluster and attach ipyrad-cli cluster-id
                self.start_ipcluster()
                self.ipyclient = self.wait_for_connection()

            # neither auto or ipyclient entered, we'll still look for default
            # profile running ipcluster and raise Error if none found.
            else:
                self.ipyclient = self.wait_for_connection()                

            # print cluster stats at this point
            # self.widget.close()
            self.get_cluster_info()

            # before running any jobs store engine pids for hard shutdown
            self.store_pids_for_shutdown()

            # run the job
            if not dry_run:
                self.tool._run(ipyclient=self.ipyclient, **self.rkwargs)
                   
        # print the error and cleanup
        except KeyboardInterrupt:
            print("\n{}Keyboard Interrupt by user\n".format(self.spacer))

        # error on remote engine.
        # print error and optionally print trace.
        except ipp.RemoteError as inst:
            msg = [
                "{}Encountered an Error.".format(self.spacer),
                "{}Message: {}: {}".format(
                    self.spacer, inst.args[0], inst.args[1])
            ]
            if not self.tool._cli:
                msg.append(
                    "{}Use debug flag (-d) for full code traceback."
                    .format(self.spacer)
                )
            print("\n" + "\n".join(msg))
            iptrace = inst.traceback

        # other errors not raised on remote engines
        # print error and always print trace
        except Exception as inst:
            msg = [
                "{}Encountered an Error.".format(self.spacer),
                "{}Message: {}".format(self.spacer, inst),
            ]
            print("\n" + "\n".join(msg))
            # get formatted traceback string
            exc_type, exc_value, exc_traceback = sys.exc_info()
            iptrace = "".join(traceback.format_exception(
                exc_type, exc_value, exc_traceback))

        # cancel/kill any unfinished jobs and shutdown hub if 'auto=True'
        finally:           
            self.cleanup()
            
            # print traceback and exit if CLI, just print if API
            if not self.tool._cli:
                if iptrace:
                    print(iptrace)
            else:
                SystemExit(1)



    def cleanup(self):
        "Cancel or kill unfinished jobs and shutdown hub if auto=True"
        try:
            # can't close client if it was never open
            if self.ipyclient:

                # Interrupt: send SIGINT (2) to all engines if any engines
                try:
                    self.ipyclient.abort()
                    time.sleep(1)
                    for eid, pid in self.tool.ipcluster["pids"].items():
                        if self.ipyclient.queue_status()[eid]["tasks"]:
                            os.kill(pid, 2)
                    time.sleep(1)
                except ipp.NoEnginesRegistered:
                    pass

                # Cleanup: purge memory so we can reuse the Client
                if not self.ipyclient.outstanding:
                    self.ipyclient.purge_everything()
                else:
                    self.auto = True
                    self.update_message(
                        "Error: ipcluster shutdown and must be restarted")

                # Shutdown the hub if it was auto-launched or broken
                if self.auto:
                    self.ipyclient.shutdown(hub=True, block=False)
                    self.ipyclient.close()
                    if self.show_cluster:
                        self.update_message(
                            "\n{}Parallel connection closed."
                            .format(self.spacer))
                        time.sleep(0.5)
        
            # close the cluster info
            # self.widget.close()

        except Exception as inst2:
            print("warning: error during shutdown:\n{}".format(inst2))




##############################################################
def detect_cpus():
    """
    Detects the number of CPUs on a system. This is better than asking
    ipyparallel since ipp has to wait for Engines to spin up.
    """
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if os.sysconf_names.get("SC_NPROCESSORS_ONLN"):
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if os.environ.get("NUMBER_OF_PROCESSORS"):
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default



def cluster_info(ipyclient):
    """ reports host and engine info for an ipyclient """    
    # get engine data, skips busy engines.
    hosts = []
    for eid in ipyclient.ids:
        engine = ipyclient[eid]
        if not engine.outstanding:
            hosts.append(engine.apply(socket.gethostname))

    ## report it
    hosts = [i.get() for i in hosts]
    hostdict = {}
    for hostname in set(hosts):
        hostdict[hostname] = hosts.count(hostname)
    hpairs = [
        # "<i>{}</i>: {} cores".format(key, val) for 
        "{}: {} cores".format(key, val) for 
        (key, val) in hostdict.items()
    ]
    print("Parallel connection | {}".format(" | ".join(hpairs)))


## GLOBALS AND EXCEPTIONS
NO_IPCLUSTER_CLI = """\
    No ipcluster instance found. This may be a problem with your installation
    setup, or it could be that the cluster instance isn't firing up fast enough.
    This most often happens on cluster nodes. One solution is to launch
    ipcluster by hand and then pass the `--ipcluster` flag to ipyrad. See
    the docs for more info: http://ipyrad.rtfd.io/HPC_script.html
    """
NO_IPCLUSTER_API = """
    No ipcluster instance found. See documentation for the proper way to set 
    up an ipcluster instance when running the ipyrad Python API. In short, 
    you must run 'ipcluster start' to initiate a local or remote cluster. 
    Also, if you changed the 'profile' or 'cluster_id' setting from their 
    default values you must enter these into the Assembly.ipcluster dictionary.
    """




# #!/usr/bin/env python

# "Parallelization with ipyparallel following ipyrad framework"

# from __future__ import print_function
# try:
#     from StringIO import StringIO
# except ImportError:
#     from io import StringIO

# import os
# import sys
# import time
# import socket
# import subprocess
# from random import getrandbits

# import ipyparallel as ipp
# from tetrad.utils import TetradError


# class Parallel(object):
#     """
#     Connect or launch ipcluster and wrap jobs running on Client engines so 
#     that engines can be interrupted or killed with a pleasant cleanup.
#     """
#     def __init__(self, tool, rkwargs, ipyclient=None, show_cluster=True, auto=False):

#         # the tool with a ._run() func and its run kwargs to be parallelized
#         self.tool = tool
#         self.rkwargs = rkwargs

#         # parallel client connect or launch params
#         self.ipyclient = ipyclient
#         self.show_cluster = show_cluster
#         self.auto = auto

    
#     def start_ipcluster(self):
#         """
#         The name is a unique id that keeps this __init__ of ipyrad distinct
#         from interfering with other ipcontrollers. Run statements are wrapped
#         so that ipcluster SHOULD be killed on exit.
#         """
#         # use random num for to cluster_id
#         rand = getrandbits(32)
#         self.tool.ipcluster["cluster_id"] = "ipp-{}".format(rand)

#         # if engines=="MPI" then add --ip arg to view all sockets  
#         iparg = ("--ip=*" if "MPI" in self.tool.ipcluster["engines"] else "")

#         # make ipcluster arg call
#         standard = [
#             "ipcluster", "start",
#             "--daemonize", 
#             "--cluster-id={}".format(self.tool.ipcluster["cluster_id"]),
#             "--engines={}".format(self.tool.ipcluster["engines"]),
#             "--profile={}".format(self.tool.ipcluster["profile"]),
#             "--n={}".format(self.tool.ipcluster["cores"]),
#             "{}".format(iparg),
#         ]
                   
#         # wrap ipcluster start
#         try:
#             subprocess.check_call(
#                 standard,
#                 stderr=subprocess.STDOUT,
#                 stdout=subprocess.PIPE)

#         # if cluster with THIS ID is running then kill it and try again
#         except subprocess.CalledProcessError:
#             subprocess.check_call([
#                 "ipcluster", "stop", 
#                 "--cluster-id", self.tool.ipcluster["cluster_id"],
#             ], 
#                 stderr=subprocess.STDOUT,
#                 stdout=subprocess.PIPE,
#             )

#             # after waiting try again to start it
#             time.sleep(3)
#             try:
#                 subprocess.check_call(
#                     standard, 
#                     stderr=subprocess.STDOUT,
#                     stdout=subprocess.PIPE)

#             # if fails again then report it
#             except subprocess.CalledProcessError as inst:
#                 print(inst)
#                 raise

#         except Exception as inst:
#             sys.exit("Error launching ipcluster for parallelization:\n({})\n"
#                 .format(inst))



#     def wait_for_connection(self):
#         """ 
#         Creates a client to view ipcluster engines for a given profile and 
#         returns it with at least one engine spun up and ready to go. If no 
#         engines are found after nwait amount of time then an error is raised.
#         If engines==MPI it waits a bit longer to find engines. If the number
#         of engines is set then it waits even longer to try to find that number
#         of engines.
#         """
#         # save stds for later, hide here to prevent ipp enforced print()
#         save_stdout = sys.stdout 
#         save_stderr = sys.stderr
#         sys.stdout = StringIO()
#         sys.stderr = StringIO()

#         # wrapped search for ipcluster
#         try: 
#             args = {
#                 "profile": self.tool.ipcluster["profile"],
#                 "timeout": self.tool.ipcluster["timeout"],
#                 "cluster_id": self.tool.ipcluster["cluster_id"],            
#             }
#             ipyclient = ipp.Client(**args)

#             # restore std printing now that Client print statement has passed
#             sys.stdout = save_stdout
#             sys.stderr = save_stderr
#             print("establishing parallel connection:")

#             # allow time to find the connection; count cores to break
#             for _ in range(6000):

#                 # how many cores can we find right now?
#                 ncores = len(ipyclient)
#                 time.sleep(0.01)

#                 # are all cores found yet? if so, break.
#                 if self.tool.ipcluster["cores"]:
#                     time.sleep(0.1)
#                     if ncores == self.tool.ipcluster["cores"]:
#                         break

#                 # If MPI and not all found, break if no more found in 3 secs
#                 if self.tool.ipcluster["engines"] == "MPI":
#                     # are any cores found yet? do long wait.
#                     if ncores:
#                         time.sleep(3)
#                         if len(ipyclient) == ncores:
#                             break

#                 # if Local then if at least one core we're happy to move on.
#                 else:
#                     if ncores:
#                         time.sleep(1)
#                         break

#         except KeyboardInterrupt as inst:
#             raise inst

#         except (IOError, OSError, ipp.TimeoutError, ipp.NoEnginesRegistered):
#             raise TetradError(
#                 "\nipcluster not found, use 'auto=True' or see docs.")

#         finally:
#             # no matter what we reset the stds
#             sys.stdout = save_stdout
#             sys.stderr = save_stderr

#         return ipyclient



#     def print_cluster_info(self):
#         """ reports host and engine info for an ipyclient """    
#         # get engine data, skips busy engines.
#         hosts = []
#         for eid in self.ipyclient.ids:
#             engine = self.ipyclient[eid]
#             if not engine.outstanding:
#                 hosts.append(engine.apply(socket.gethostname))

#         ## report it
#         hosts = [i.get() for i in hosts]
#         result = []
#         for hostname in set(hosts):
#             result.append(
#                 "{}host compute node: [{} cores] on {}"
#                 .format(hosts.count(hostname), hostname))
#         print("\n".join(result))



#     def store_pids_for_shutdown(self):
#         "reset tool ipcluster dict pids dict and set with current engine pids"
#         self.tool.ipcluster["pids"] = {}
#         for eid in self.ipyclient.ids:
#             engine = self.ipyclient[eid]
#             if not engine.outstanding:
#                 pid = engine.apply(os.getpid).get()
#                 self.tool.ipcluster["pids"][eid] = pid



#     def wrap_run(self):
#         """
#         Takes an analysis tools object with an associated _ipcluster attribute
#         dictionary and either launches an ipcluster instance or connects to a 
#         running one. The ipyclient arg overrides the auto arg.
#         """
#         try:
#             # check that ipyclient is running by connecting (3 seconds tries)
#             if self.ipyclient:
#                 for i in range(3):
#                     if len(self.ipyclient):
#                         break
#                     else:
#                         time.sleep(1)
#                 assert len(self.ipyclient), "ipcluster not connected/running."

#             # launch ipcluster and get the parallel client with ipp-{} id
#             elif self.auto:
#                 # set default to 4
#                 if not self.tool.ipcluster["cores"]:
#                     self.tool.ipcluster["cores"] = 4

#                 # start ipcluster and attach ipyrad-cli cluster-id
#                 self.start_ipcluster()
#                 self.ipyclient = self.wait_for_connection()

#             # neither auto or ipyclient entered, we'll still look for default
#             # profile running ipcluster.
#             else:
#                 self.ipyclient = self.wait_for_connection()                

#             # print cluster stats at this point
#             if self.show_cluster:
#                 self.print_cluster_info()

#             # before running any jobs store engine pids for hard shutdown
#             self.store_pids_for_shutdown()

#             # run the job
#             self.tool._run(**self.rkwargs, ipyclient=self.ipyclient)

#         # print the error and cleanup
#         except KeyboardInterrupt:
#             print("\nKeyboard Interrupt by user\n")

#         except Exception as inst:
#             print("\nEncountered an error:\n{}\n".format(inst))
#             raise

#         # cancel/kill any unfinished jobs and shutdown hub if 'auto=True'
#         finally:           
#             self.cleanup()


#     def cleanup(self):
#         "Cancel or kill unfinished jobs and shutdown hub if auto=True"
#         try:
#             # can't close client if it was never open
#             if self.ipyclient:

#                 # Interrupt: send SIGINT (2) to all engines if any engines
#                 try:
#                     self.ipyclient.abort()
#                     time.sleep(1)
#                     for eid, pid in self.tool.ipcluster["pids"].items():
#                         if self.ipyclient.queue_status()[eid]["tasks"]:
#                             os.kill(pid, 2)
#                     time.sleep(1)
#                 except ipp.NoEnginesRegistered:
#                     pass

#                 # Cleanup: purge memory so we can reuse the Client
#                 if not self.ipyclient.outstanding:
#                     self.ipyclient.purge_everything()
#                 else:
#                     self.auto = True
#                     print("\nerror: ipcluster shutdown and must be restarted")

#                 # Shutdown the hub if it was auto-launched or broken
#                 if self.auto:
#                     self.ipyclient.shutdown(hub=True, block=False)
#                     self.ipyclient.close()
                
#         except Exception as inst2:
#             print("warning: error during shutdown:\n{}".format(inst2))
