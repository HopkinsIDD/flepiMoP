# Provisioning AWS EC2 instance

{% hint style="warning" %}
This page, along with the other AWS run guides, are not deprecated in case we need to run `flepiMoP` on AWS again in the future, but also are not maintained as other platforms (such as longleaf and rockfish) are preferred for running production jobs.
{% endhint %}

### Signing in to AWS Management Console&#x20;

Click on below:

{% embed url="https://aws.amazon.com/console/?nc1=h_ls" %}

Sign in as IAM user with your given Accound ID, username and Password

<figure><img src="../../.gitbook/assets/スクリーンショット 2022-10-14 10.07.24.png" alt=""><figcaption><p>Sign in as IAM user</p></figcaption></figure>

&#x20;Then the next view appears, check "regeon" as "Oregon" by default and "user@Accond ID" as you expeced.

<figure><img src="../../.gitbook/assets/スクリーンショット 2022-10-14 10.10.02.png" alt=""><figcaption><p>Console Home</p></figcaption></figure>

If you have already accessed AWS console, these kinds of view can be seen. In the case select "EC2" to go to "EC2 Dashboard"(if not, skip it).

### EC2 Dashboard

In this EC2 Dashboard, we can maintain the EC2 boxes from creation to deletion. In this section, how to create an EC2 instance from the AMI image which has already been registered is shown.

<figure><img src="../../.gitbook/assets/スクリーンショット 2022-10-14 10.22.56.png" alt=""><figcaption><p>EC2 Dashboard</p></figcaption></figure>

Select "Images>AMIs" in the right pane(Navigation pain),&#x20;

Select an AMI name which name is "IDD Staging AMI" in the "Amazon Machine Images (AMIs)"  by clicking the responding AMI checkbox on the left, then push the  "Launch instance from AMI" button (colored in orange).

<figure><img src="../../.gitbook/assets/スクリーンショット 2022-10-14 10.31.32.png" alt=""><figcaption><p>Select an AMI Image among Amazon Machine Images(AMIs)</p></figcaption></figure>

### Launch an instance

To create an EC2 instance, fill out the items as below (example):

* Name and tags
  * input an appropriate name (e.g., _"sample\_box01"_)
* Application and OS image
  * check whether _"AMI from catalog"_ is _"IDD Staging AMI" (for example; select one as you want)_&#x20;
* Instance type
  * as you selected by drop-down list(e.g., _m5.xlarge_)
* Key pair(login)&#x20;
  * you can generate new key pair if you want to connect to the instance securely (by clicking "Create new key pair" on the right side), but usually select "ams\__ks\_ED25519\__keypair" by drop-down list so that you can be helped when local client setup (recommended).
    * In case that you use your own key, you will be the only person to log in, of course. you should be careful of handling key management.&#x20;
* Network settings (push the button "Edit" on the right to extend configuration; see below)
  * VPC - required
    * select _"HPC VPC"_ item by drop-down menu
  * Subnet
    * select "HPC Public Subnet among _"us-west-2\*"_&#x20;
  * Firewall (security groups)
    * select "_Select existing security grous"_ toggle, then
    * Common security groups
      * select "dvc_usa" and "dvc\__usa2" by drop-down menu

<figure><img src="../../.gitbook/assets/スクリーンショット 2022-10-14 12.48.08.png" alt=""><figcaption><p>Network settings </p></figcaption></figure>

*   Advanced details

    * "EC2S3FullAccess" should be setected in IAM instance profile, but to do it an authentication (IAM role or policy) must be set on to the working IAM account



    <figure><img src="../../.gitbook/assets/スクリーンショット 2022-11-23 午前11.58.58.png" alt=""><figcaption><p>Advanced details</p></figcaption></figure>



then push "Launch Instance" button which is located at the bottom right side of the screen&#x20;

<figure><img src="../../.gitbook/assets/スクリーンショット 2022-10-14 12.50.32.png" alt=""><figcaption><p>Launch Instance in Summary</p></figcaption></figure>

<figure><img src="../../.gitbook/assets/スクリーンショット 2022-10-14 12.51.59.png" alt=""><figcaption><p>When in Success</p></figcaption></figure>

