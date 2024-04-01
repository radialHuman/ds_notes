## [Introduction]()
Release date : 
### Idea
- Create stack on existing template and delete it
- Write template with yaml, and updating stack with template
- CF reference
- parameters in templates
- Metadata and mapping
- Conditional 
- Define output
- Change sets
- cmd line 
- Required
    - AWS intermediate knowldegde
    - yaml basics
- There is a intermediate to advanced course too beyond this

### Details
#### 2
- To model and provision AWS service in an automated way
- AWS CF is free
- resources will cost
- Infra as code is created and can be version controlled
- Either by using 
    - AWS mgmt console 
    - AWS CLI
    - AWS SDK (python)
- To aovid human error while provisioning as it is testable like a code
- Tempalate can be reused and shared (like python vs excel)
- Recreating and erasing them is easy and quick there by it safe time and money
- components
    - templates : blue print of infra defining resources paramters conditions and output. usually in json or yaml type.
    - stacks 
        - single unit with multiple aws resources to manage
        - stack can be updted deleted easily
        - gets details of resources from template
        - a stack can have one tmeplate but a template can be used to create multiple stacks
    - change sets : 
        - Its a summary of all the change smade to the stack
        - It allows to see unexpected imapct of the changes

#### 3
- A sample template is prepared in resources which will be used to work and create stack on
- Details of the template will be discussed later
- AWS CF -> create stack -> template is ready
    - use samepl temp can provide builtin temp as per usecase
    - create using designer is drag and drop 
    - both are easier hence ignore in this course
- upload the template from local , will automatically create s3 bucket for future use
- Stack name unique
- parameters will be empty and will be discussed later
- Configure stack options
    - taggable resources in tag and value
    - permission using IAM also can be specified
    - THere are many more advanced options but for now skip them all
- Review and submit
- This leads to stack details with
    - Events
    - Resources
    - info
    - output
    - parameters
    - template
- all the resources can be checked manually

#### 4
- Deleteing the resources created
- go to stack list
- no need to manually delete resource 1/1
- select the stack from the stack list to be deleted
- once the stack details are visible
    - action -> delete stack
- Events will show deleting in progress
- manually check resource 1/1
- stack info will get updated
- nothing in stack list
- can be found in deleted

#### 5
- Activity 1
- use the other yaml in resources and upload in new stack
- SKIPPING


### Resource
- 

### misc
 
---
## [Templates and Resources]()
Release date : 
### Idea
- Aim:
    - Understanding temp anatomy and writing one from scratch in yaml
    - updating stack after editing its temp
    - How to use AWS ref doc to define different aws resources
    - intrinsic function and ref function
    - why some resources in stack gets replaced during update
    - how CF orders creation and deletion of resouce in stack
    - dependOn attribute to define explicit dependencies

### Details
#### 2 : Temp anatomy
- Has 9 major sections
- Resources is the most important one
- sections can be in any order but there is aorder recommended
    - since there can be a reference to something in previous section
- order
    1. format version
        - optional, date
        - constant values is 2010-09-09 as of 30/03/24
        - AWSTemplateFormatVersion:
    2. description
        - optional, string
        - can be anything you want to convey regarding the temp
    3. metadata
        - optional
        - json or yaml objects to give more info about the file
        - Values form this section can be referenced later in the temp
        - ex : AWS::CLoudFormation::Interface key for ordering and grouping parameters
    4. parameters
        - optional
        - passes values at runtime while creating and updating stacks
        - increases reuseablility of the temp
        - Values can be referenced from Resource or Output section
    5. mappings
    6. conditions
        - optional
        - conditions when met will create resources are defined here
        - ex : depending on the env prod or dev, resources will be created accordingly
    7. transform
        - Optional
        - new  section
        - Specifies AWS serverless applciation model version
        - if something like lambda function is sued then this section will be useful
    8. resources
        - Required
        - prop and resource details
    9. outputs
        - optional
        - define outputs like dns name or load balancers
        - output then can be visible in console or cli
        - advanced use : create cross-stack references and pass values between nested stacks

#### 3
```yaml
AWSTemplateFormatVersion: 2010-09-09
Description: testing writting template
Resources: #whatever comes under it will be tabbed
 # every resource should have logical ID which is alpha numeric and unique to the template
    WebServerInstance: # these are logical IDs which can be anything but comprehensibe and unique for each resource
        Type: AWS::EC2::Instance
        # this is can be found in the resource and property type documentation online, select EC2 and then all the EC2 related types will be listed, click and see its properties in yaml and json, required 
        Properties: # this is to confugure the resource
            # There are too many, can be easily fetched by extension bt here its manual
            ImageID: ami-0ds0fdf0df # provides unique Amazon machine image which can be found in launch instance of ec2
            InstanceType: t2.micro # from instance families and types
            Tag: 
            # not required, unique for resources, it not a string its Resource tag format which is k:v pairs
                -
                    key: Name
                    value: Web server
                - 
                    key : Project
                    value : step by step learning
``` 
- Upload it with a new stack

#### 4 Editing a stack
- Adding a new resource
```yaml
AWSTemplateFormatVersion: 2010-09-09
Description: testing writting template
Resources: 
    WebServerInstance: 
        Type: AWS::EC2::Instance
        types will be listed, click and see its properties in yaml and json, required 
        Properties: 
            ImageID: ami-0ds0fdf0df 
            InstanceType: t2.micro 
            Tag: 
                -
                    key: Name
                    value: Web server
                - 
                    key : Project
                    value : step by step learning
    WebServerSecurityGroup:
        Type: AWS::EC2::SecurityGroup
        Property: # get it form https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ec2-securitygroup.html
            GroupDescription: Sec group for web service #required
            # GroupName: String #not required, auto generated unique id
            # SecurityGroupEgress: 
                # - Egress
            SecurityGroupIngress:  # not required, but its a custom format, so details from doc
                - # copy details from doc since its about security, its out of scope
            Tag: 
                -
                    key: Name
                    value: Web server sec group
            # VpcId: String
```
- Save the template
- To update the stack go to the CF gui 
    - Action -> update stack -> replace current template
    - Similar to new template upload upload the updated one from s3/local
    - make the same configuration as done while building the OG one
    - in review change set preview check for the new resources added
    - review and update it
- When the update is completed then check the resources 
- But the new resource si not conencted to the OG resource EC2

#### 5 Linking two resources
- Intrinsic function help in linking two resources
- Bulit in groups of functions to manage stack
- used for assining prop during runtime
- most common one is ref function
    - used to retuen value of a resoucre or parameter
    - using the logicla id of the resource, ref fucntion gets value to be used as reference to that resource
    - The returned value are in AWS CF resource and propert type reference doc
    - In any resoruce doc page look for return value tab on the right
        - ex :https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ec2-securitygroup.html#aws-resource-ec2-securitygroup-return-values
    - Where can this information be palced in another resource in outr case ec2
        - GO to ec2 instance resource type
        - there in yaml we find two places
            - SecurityGroupIds: 
                - String #  Array of String
            - SecurityGroups: 
                - String # ignore
        - to make changes in the yaml file
```yaml
WebServerInstance: 
        Type: AWS::EC2::Instance
        types will be listed, click and see its properties in yaml and json, required 
        Properties: 
            ImageID: ami-0ds0fdf0df 
            InstanceType: t2.micro 
            # this s where the new lines for reference will go
            SecurityGroupIds: # this tag is from doc, depends on the resource type for existence
            # this is one way of typing the ref function
                - 
                    Ref: WebServerSecurityGroup
            # there is another way which will come later
            Tag: 
                -
                    key: Name
                    value: Web server
                - 
                    key : Project
                    value : step by step learning
```
- Once it supdated, it should be updated in the stack in gui, the same way as in previous section
- Result can be seen in EC2 instance resource
    - look for unning instance
    - see its desciprtion
    - Details from securitygroup will be visible
- Here the previous version of ec2 without ref will get terminated
- and a new version will be updated

#### 6 Stack update with replacement
- When a resource is udpated, old one is terminated and removed later and replaces it with new instance
- This is visible in Events log of CF
- Even though adding sec group to running ec2 instance is easy the resource of ec2 insatnce is deleted and replaced with a new one
- More info of this can be found in https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-updating-stacks-update-behaviors.html#update-some-interrupt
    - Default: Amazon EC2 uses the default security group.
    - Required: No
    - Type: Array of String
    - Update requires: Some interruptions
- Since no vpc was given , it created a new one
- Deletec stack to clean resources
    - but there can be dependcies, so ordering them is important

#### 7 Ordering Resource creation
- How to order resources and
- How to define custom dependencies
- Implicit ordering (when there is order automatically)
    - When there is no depdnencies between resources, they are created and deleted parallely
    - Like in the fsecond video, without refrence in EC2 instance
    - When there is depdnency i.e. ec2 has reference from security group
        - Sec group is created before and then ec2 instance gets started
    - while deleting it (its reverse)
        - first ec2 will be taken away then sec group
- Explicit ordering (when the order is user enforced)
    - ```yaml
        AWSTemplateFormatVersion: 2010-09-09
        Description: testing writting template
        Resources: 
            WebServerInstance: 
                Type: AWS::EC2::Instance
                types will be listed, click and see its properties in yaml and json, required 
                Properties: 
                    ImageID: ami-0ds0fdf0df 
                    InstanceType: t2.micro 
                    Tag: 
                        -
                            key: Name
                            value: Web server
                        - 
                            key : Project
                            value : step by step learning
            WebServerSecurityGroup:
                Type: AWS::EC2::SecurityGroup
                Property: aws-resource-ec2-securitygroup.html
                    GroupDescription: Sec group for web service 
                    SecurityGroupIds: 
                        - 
                            Ref: WebServerSecurityGroup
                    Tag: 
                        -
                            key: Name
                            value: Web server sec group
                    
            # creating a new resource sqs with explicit depdndency in ec2
            SQSQueue:
                Type: AWS::SQS::Queue
                # if its stopped here then all 3 resources will be created and deleted in ||le
                # to force creation os sqs after ec2 the following helps
                # depndsON must be at type indentation level
                DependsOn: # multiple things can be added  ??? will that also be in order
                    - WebServerInstance
        ```

#### 8 Demo activity
- Part 1
    1. Define VPC resource
    2. Add subnet to VPC which maps public IP to EC2 instance launched inside it
    3. add route table in vpc associate ti with VPC
    4. Attach internet gateway to VPC
    5. Add a route in table for internet access via your internet gateway
    6. Define Ec2 instance in public subnet
    7. The create stack form this temp
- Part 2
    1. Edit temp : add sec group which has ingress rule to allow ping ICMP protocol
    2. Attach this sec group to ec2 instance
    3. Update stack with edited temp
    4. Test ec2 by oing public ip address

#### 9 Demo answer
- Adding ony required property
- Not sure about the resources and details so help is taken from the answer
- Part 1
```yaml
    AWSTemplateFormatVersion: 2010-09-09
    Description: demo 2
    Resources:
    # 1
        VPCResource:
            Type: AWS::EC2::VPC
            Description : VPC resource
            Property: 
                CidrBlock: 10.10.10.0/16
            Tags:
                -
                    Key : Name
                    Value : VPC
    # 2
        SubnetResource:
            Type: AWS::EC2::Subnet
            Property : 
                CidrBlock: 10.10.10.0/24
                VpcId: !Ref VPCResource # required, taken from answer
                MapPublicIpOnLaunch: true # for ec2
                Tag: 
                    - 
                        Key: Name
                        Value : Subnet
    # 3
        RouteTable:
            Type : AWS::EC2::RouteTable
            Property:
                VpcId: !Ref VPCResource # required, taken from answer
                Tag: 
                    - 
                        Key: Name
                        Value : RouteTable
    # 4a
        InternetGateway:
            Type : AWS::EC2::InternetGateway
            # to connect to VPC, see more info in https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ec2-vpcgatewayattachment.html
    # 4b
        VPCInternetGateway:
            Type : AWS::EC2::VPCGatewayAttachment
            Property:
                VpcId: !Ref VPCResource # required, taken from answer
                InternetGatewayId : !Ref InternetGateway 
    # 5a
        InternetRoute:
            Type: AWS::EC2::Route
            DependsOn: InternetGateway # required as its dependency, from the answer
            Property: 
                RouteTableId: !Ref RouteTable
                DestinationCidrBlock: 0.0.0.0/0
                GatewayId: !Ref InternetGateway
    # 5b
        #not sure why this is in the answer
        SubRouteTable:
            Type: AWS::EC2::SubnetRouteTableAssociation
            Properties:
                RouteTableId: !Ref RouteTable
                SubnetId: !Ref SubnetResource
    # 6
        EC2Instance:
            Type: AWS::EC2::Instance
            DependsOn:
                - InternetRoute
                - SubRouteTable
            Properties:        
                InstanceType : t2.micro
                SubnetId: !Ref SubnetResource
                ImageId: ami-q34235435345

```
- Upload
- Part 2
```yaml
        EC2Instance:
                    Type: AWS::EC2::Instance
                    DependsOn:
                        - InternetRoute
                        - SubRouteTable
                    Properties:        
                        InstanceType : t2.micro
                        SubnetId: !Ref SubnetResource
                        ImageId: ami-q34235435345
                    SecurityGroupIds:
                            - !Ref ActivitySecurityGroup
        ActivitySecurityGroup:
            Type: AWS::EC2::SecurityGroup 
            Properties:
            GroupDescription: Activity security group #not required
            VpcId: !Ref ActivityVpc
            SecurityGroupIngress:
                -
                CidrIp: 0.0.0.0/0 
                IpProtocol: icmp
                FromPort: -1
                ToPort: -1
```

### Resource
- 

### misc
 
---

## [Parameters]()
Release date : 
### Idea
- 

### Details
#### 2 Intro
- Optional section in CF template, to procvide custom values  while creating or updating stacks
- Once its defined (like a variable), it can be read from resources or output section in the same template only
- !Ref can be used to call parameters just like resources
- Type of the parameter needs to be declared, so that it can be validated before creation or update of stack
- Parameter types allowed:
    - String : "xyz"
    - Number : "123"
    - List of numbers : ["80","20"]
    - CommaDelimitedList : "test,dev,prod"
    - AWS-specific parameter types :  from aws account
        - AWS::EC2::AvailabilityZone::Name
        - AWS::EC2::AvailabilityZone::Name
        - AWS::EC2::Image::Id
        - AWS::EC2::Instance::Id
        - AWS::EC2::KeyPair::KeyName
        - AWS::EC2::SecurityGroup::GroupName
        - AWS::EC2::SecurityGroup::Id
        - AWS::EC2::Subnet::Id
        - AWS::EC2::Volume::Id
        - AWS::EC2::VPC::Id
        - AWS::Route53::HostedZone::Id
        - List<AWS::EC2::AvailabilityZone::Name>
        - List<AWS::EC2::Image::Id>
        - List<AWS::EC2::Instance::Id>
        - List<AWS::EC2::SecurityGroup::GroupName>
        - List<AWS::EC2::SecurityGroup::Id>
        - List<AWS::EC2::Subnet::Id>
        - List<AWS::EC2::Volume::Id>
        - List<AWS::EC2::VPC::Id>
        - List<AWS::Route53::HostedZone::Id>
    - SSM parameter types : from systems manager paramter store, advanced
         - https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/parameters-section-structure.html#aws-ssm-parameter-types
- Parameter constraints
    - AllowedPattern
    - AllowedValues
    - ConstraintDescription
    - Default
    - Description
    - MaxLength
    - MaxValue
    - MinLength
    - MinValue
    - NoEcho
- There can be default values to these paramters also 
- ALl this increases reuseability of the template
- To prevent hardcoding in resources and properties like instance type, username and password


#### 3 Defining parameters
- Parameters are to be added before the resource as they are called after and looked before
- Best to write the resources first and then add parameters ass required, above it
```yaml
AWSTemplateFormatVersion: 2010-09-09
Description: Sample database stack for the Parameters section
Parameters:
  DbClass: # parameter name, like resource name cna be anythin but unique
    Type: String 
    Description: RDS instance class  # optional, but will be visible in GUi
    AllowedValues: # constraint
      - db.t2.micro
      - db.t2.small
    ConstraintDescription: 'DbClass parameter can only have these values: db.t2.micro, db.t2.small' # optinal but useful
    Default: db.t2.micro 
  MasterUsername:
    Type: String 
    Description: Master username for the db instance 
    MaxLength: 10
    Default: dbadmin
    AllowedPattern: '[a-zA-Z][a-zA-Z0-9]*'
    NoEcho: true 
  MasterUserPassword:
    Type: String 
    Description: Master user password for the db instance 
    MinLength: 8
    NoEcho: true 
  MultiAZ:
    Type: String
    Description: Enable Multi-AZ?
    AllowedValues:
      - true 
      - false
    ConstraintDescription: MultiAZ parameter should be either true or false.
    Default: false 
  AllocatedStorage:
    Type: Number 
    Description: Database storage size in GB
    MinValue: 8
    MaxValue: 20
    ConstraintDescription: AllocatedStorage parameter value should be between 8 and 20.
    Default: 8
  SecurityGroupPorts:
    Type: List<Number>
    Description: 'Port numbers as a list: <web-server-port>,<database-port>'
    Default: '80,3306'
  DbSubnets:
    Type: List<AWS::EC2::Subnet::Id>
    Description: 'Db subnet ids as a list: <subnet1>,<subnet2>,...'
  VpcId:
    Type: AWS::EC2::VPC::Id 
    Description: A valid VPC id in your AWS account
Resources:
  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      VpcId: !Ref VpcId 
      GroupDescription: 'Web server instances security group'
      SecurityGroupIngress:
        - 
          CidrIp: 0.0.0.0/0
          FromPort: 
            Fn::Select: [ 0, !Ref SecurityGroupPorts ]
          ToPort: !Select [ 0, !Ref SecurityGroupPorts ]
          IpProtocol: tcp

  # Note: Please replace the value of VpcId property
  # with the VPC id of your default VPC
  DbSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      VpcId: !Ref VpcId
      GroupDescription: 'Database instances security group'
      SecurityGroupIngress:
        - 
          CidrIp: 0.0.0.0/0
          FromPort: !Select [ 1, !Ref SecurityGroupPorts ]
          ToPort: !Select [ 1, !Ref SecurityGroupPorts ]
          IpProtocol: tcp

  # Note: Please replace the value of SubnetIds property 
  # with the subnet ids of the subnets in your default VPC!
  DbSubnetGroup:
    Type: 'AWS::RDS::DBSubnetGroup'
    Properties:
      DBSubnetGroupDescription: Subnets to launch db instances into
      SubnetIds: !Ref DbSubnets

  DatabaseInstance:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceClass: !Ref DbClass  # reference parameter
      Engine: mariadb
      MultiAZ: !Ref MultiAZ
      PubliclyAccessible: true
      AllocatedStorage: !Ref AllocatedStorage 
      MasterUsername: !Ref MasterUsername
      MasterUserPassword: !Ref MasterUserPassword
      DBSubnetGroupName: !Ref DbSubnetGroup
      VPCSecurityGroups: 
        - !Ref DbSecurityGroup
      
```
- Once this is uploaded in a new stack, there will be a Parameter section in the second page where parameter name will be displayed
- Only by entering allowed values for these parameters manully can the stack be created else error

### Resource
- 

### misc
 
---
