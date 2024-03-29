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
- 

### Details
- 

### Resource
- 

### misc
 
---
