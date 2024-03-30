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
    WebServerInstance: # this is a logical id?
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

### Resource
- 

### misc
 
---
