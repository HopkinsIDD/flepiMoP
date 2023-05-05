import boto3
import datetime
import subprocess
import re
import time
 

bucket = "idd-inference-runs"  # 'idd-inference-runs'
bucketdest = "idd-inference-runs-backup"
# vault = "backup-20230428" # no need to set up a vault because files must be converted to "glacier deep archive", not to "glacier"
destinationdir ="./backup" # local


s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix="", Delimiter="/")  # needs paginator cause more than 1000 files

to_prun = []
# folders:
to_gdalist = [] # prefix to gda list 

def check_gda(paginator, bucket, prefix):  # precheck to avoid glacier deep archives
    # print("bucket:",bucket,"prefix:",prefix)
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for file in page['Contents']:
            if file['StorageClass'] == "DEEP_ARCHIVE":
                print("\tGlacier Deep Archive file found on ", file['Key'],", not append to the target list")
                return True
    return False

print("Searching what to delete or keep: ")
for page in pages:
    for cp in page["CommonPrefixes"]:
        prefix = cp["Prefix"]
        # Find the date, not always this easy: prefix.split('-')[1].replace('/','')
        m = re.search(r"\d", prefix)
        try:
            timestamp = prefix[m.start() : m.start() + 15]
            rundate = datetime.datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
            if rundate < datetime.datetime.now() - datetime.timedelta(weeks=48):
                print(f"- Will be the targets in {prefix}, rundate {rundate}")
                if (check_gda(paginator, bucket, prefix)): # wanted to pass paginator directly but so far search cond should not be appened later?
                  #  return True if contais obj.storage_class == "DEEP_ARCHIVE"
                    to_gdalist.append(prefix)
                    pass
                else:
                   to_prun.append(prefix)
            else:
                print(f"- NOT pruning {prefix}, rundate {rundate}")
        except ValueError: # something like  "USA-USA-20221129Tnoseed"
            print(f"ValueError arose, ignore {prefix}")
            pass
        except AttributeError: # something like "USA-NA"
            print(f"AttributeError arose, ignore {prefix}")
            pass


def perform_targz(to_prun, do_it_for_real=False):
    dry_run_str = "--dryrun" # remains as is
    if do_it_for_real == True:
        print("this is not a dry run, waiting 3 second so you reconsider")
        time.sleep(3)
        dry_run_str = ""
    with open("torun2targz.sh", "w") as script_file, open("torun2deletes3files.sh", "w") as script2_file, open("push2gdalist", "w") as gdalist:
        for run in to_prun:
            print(f"target directory: {run}...", end="")
            command = f"aws s3 sync  s3://{bucket}/{run} {destinationdir}/{run}"
            command1a = f"aws s3 cp  s3://{bucket}/{run[:-1]}-copy.sh {destinationdir}"
            command1b = f"aws s3 cp  s3://{bucket}/{run[:-1]}-runner.sh {destinationdir}"
            command1c = f"aws s3 cp  s3://{bucket}/{run[:-1]}.tar.gz {destinationdir}"
            # aws s3 cp --recursive --exclude "*" --include "CA-20210702T142029-*.sh" s3://idd-inference-runs/ ./backup
            # above 1-line command cannot be used because searching is too slow!
            print(f">>> {command}")
            print(f"echo copying {run}", file=script_file)
            command2 = f"tar czvf  {destinationdir}/{run[:-1]}.taz -C {destinationdir} {run} {run[:-1]}-copy.sh {run[:-1]}-runner.sh {run[:-1]}.tar.gz"
            command3 = f"aws s3 cp {destinationdir}/{run[:-1]}.taz s3://{bucketdest}/{run[:-1]}.taz --storage-class DEEP_ARCHIVE"
            command4 = f"rm -fr {destinationdir}/{run} {destinationdir}/{run[:-1]}-*.sh {destinationdir}/{run[:-1]}.tar.gz"
            print(
            #    f"{command} && cd {destinationdir} && {command2} && rm -fr {run} || {{ echo 'failed for {run} !' ; exit 1; }}",
                f"{command} && {command1a} && {command1b} && {command1c} &&  {command2} && {command3} && {command4}  || {{ echo 'failed for {run} !' ; exit 1; }}",

                file=script_file,
            )
            # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            # process.wait()
            # print(f"Done, return code is {process.returncode} !")
            # if process.returncode != 0:
            #    raise ValueError(f"STOPPING, aws s3 rm failed for {run} !")
            command_d1 = f"aws s3 rm  s3://{bucket}/{run} --recursive"
            command_d1a = f"aws s3 rm  s3://{bucket}/{run[:-1]}-copy.sh"
            command_d1b = f"aws s3 rm  s3://{bucket}/{run[:-1]}-runner.sh"
            command_d1c = f"aws s3 rm  s3://{bucket}/{run[:-1]}.tar.gz"
            print(f">>> {command}")
            print(f"echo deleting {run}", file=script2_file)
            print(
                f"{command_d1} && {command_d1a} && {command_d1b} && {command_d1c}  || {{ echo 'failed for {run} !' ; exit 1; }}",

                file=script2_file,
            )
        for i in to_gdalist:
            print(f"{bucket}/{i}", file=gdalist,)

            

print("I'll perform the deletion, this is dangerous...")
if input("... Do you wish to continue? [yes/no] ") == "yes":
    do_it_for_real = True
    # do_it_for_real = False
    if do_it_for_real:
        if input("... NOT A DRY RUN, is that really ok ? [yes/no] ") == "yes":
            perform_targz(to_prun=to_prun, do_it_for_real=do_it_for_real)
        else:
            print("wise choice, abording")
            exit()
    else:
        perform_targz(to_prun=to_prun, do_it_for_real=do_it_for_real)


else:
    print("wise choice, abording")
    exit()

# files
# for page in pages:
#  for obj in page['Contents']:
#      print(obj['Key'], obj['LastModified'])

# filter and aws s3 rm --recursive --exclude '*' --include '*/intermediate/*' --include '*/chimeric/*' s3://idd-inference-deletetest/USA-20210903T192833/
