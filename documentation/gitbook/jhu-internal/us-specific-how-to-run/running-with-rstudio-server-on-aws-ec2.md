# Running with RStudio Server on AWS EC2

{% hint style="warning" %}
This page, along with the other AWS run guides, are not deprecated in case we need to run `flepiMoP` on AWS again in the future, but also are not maintained as other platforms (such as longleaf and rockfish) are preferred for running production jobs.
{% endhint %}

## Introduction

As a computational environment, you can use an RStudio Server integrated AWS EC2 instance for either your personal space or shared usages among multiple users, via GUI as well as CLI using ssh.  The EC2 instance-type was selected to be appropriate one for running programs as of now (2023/1), in views of both computational resources and finances, so that you can use a cloud-based computing environment which you can access with GUI including  from Web, without any difficulties to set it up. The details hereinafter may be able to change.

## Versions

&#x20;The current installed versions of software or additional information related to AWS EC2 are as follows:

* R/RStudio Server
  * R version: 4.2.2
  * RStudio Server version: v2022.07.02+576
* AWS EC2 instance configurations
  * instance-type: r6i.4xlarge (16 cores, 128GB mem ;
  * Storage: 2TB x 1 (gp3)
  * OS: ubuntu 22.04 (Jammy)

## Provisioning a server on AWS EC2

To be written/ Talk to someone who would be able to do that.

* EC2 instance initialization with specfic AMI
* Configured network related including ports openings
* registration of the user in the EC2 instance
* Configuring shared directory via SMB and accoun ;

## Starting an EC2 instanc ;

The procedure is same as a normal ec2 instance starting. One way is to select the ec2 instance and start it in EC2 Management Console ;

Once the instance started, RStudio Server can be accessed without invoking manually ;

## Accessing RStudio Server

By default RStudio Server runs on port 8787 and accepts connections from all remote clients. After invoking an EC2 box you should therefore be able to navigate a web browser to the following address to access the server:

> http://\<ip-addr>:8787/

Then the authentication dialog will be shown, try to log in by inputting your username and password which are already registered in the box and pushing the "Sign In " button:

<figure><img src="../.gitbook/assets/スクリーンショット 2023-01-10 午後12.22.33.png" alt=""><figcaption><p>Authentication dialog</p></figcaption></figure>

RStudio view can be appeared as below:

<figure><img src="../.gitbook/assets/スクリーンショット 2023-01-10 午後12.27.24.png" alt=""><figcaption><p>RStudio view</p></figcaption></figure>

## Accessing the Linux server(Ubuntu) using RDP

To access the linux server with GUI, RDP software can be applicable, in addition to the usual way, via ssh with command line.



### Using Windows

By using "Remote Desktop Connection" app in Windows, you can log in to the Linux server box from remote environment.

### Using Mac

For Mac users, the below RDP software is recommented:

#### **Microsoft Remote Desktop**

[https://apps.apple.com/us/app/microsoft-remote-desktop/id1295203466](https://apps.apple.com/us/app/microsoft-remote-desktop/id1295203466)

## Accessing the shared space on Linux server

As a shared space, the directory named:

```
/home/shared
```

is deployed among multiple server boxes using EFS(Elastic File System) which covers NFSv4 protocol. ;

## Accessing the shared space on Linux server using Samba(obsolete)

### Common

In the linux box, Samba(SMB) service has been on for file exchanging by default. The area in which can be readable and writable under the specific user privileages is:

```
/home/share
```

When accessing the area via SMB, you can input username and its password in a dialog window which will be shown. The username is:

```
smbshare
```

&#x20;(ask password for above user in advance, if you want to access via SMB)

### Using Windows

By inputting the form such as `\\<ip-addr>\share` in Windows Explorer

### Using Mac

From Finder you can access the shared space using SMB.

1. From Finder Menu, choose "MOVE" then "Connect to Server"
2. When a dialog appears, fill username and password out as a registered user ;
3. After pushing "connect" button, the designated area will be shown in Finder if no errors happen ;

<figure><img src="../.gitbook/assets/スクリーンショット 2023-01-09 午後2.06.45.png" alt=""><figcaption><p>Dialog window (in Japanese form; this will be changed according to yours OS's locale)</p></figcaption></figure>



### Notes

When you are inside of the university networks, e.g. in labs or in office, you will not access to the server box with SMB because the networks may be blocking the ports related to the services.

If you are using MAC as a local pc, there is a workaround to avoid the situation but for Windows it has not been clear there is a solution (now under investigation). If you want to know the related information, currently even for Mac user only though, please try to make a contact. In case of a Windows user, I recommend using "Local devices and resources" setting of Remote Desktop Connection.\


