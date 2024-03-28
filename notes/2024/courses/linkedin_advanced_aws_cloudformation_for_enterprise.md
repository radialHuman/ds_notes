[github](https://github.com/LinkedInLearning/advanced-aws-cloudformation-for-enterprise-4315140)

## [intro]()
Release date : 
### Idea
- Instead of building infra in AWS using gui and by setting up config 
- use infra as code in CF
- create dev env
- create reuseable components
- conditional logic
- manage resources across account
- automate template

### Details
- Requirements
    - AWS basics
    - CF funcdamentals
    - AWS : deployment, provisioning and automation
    - AWS : storage and data mgmt
- Tools
    - aws cli
    - yaml [cheatsheet](https://quickref.me/yaml)
    - vs code
    - python 3.7 with pip
- aws account with root user
- IAM permission with limited acces AWS : security
- AWS free tier

### Resource
- [github](https://github.com/LinkedInLearning/advanced-aws-cloudformation-for-enterprise-4315140) 

### misc
 
---

## [dev env]()
Release date : 
### Idea
- Before writing the yaml file a dev en needs to be setup using
    - awscli
    - cfn-lint by AWS in vs code
        - catches errors
        - enforces good pracice
        - resource lookup
        - direct doc reference
    

### Details
#### 1
- installation
    - pip install cfn-lint
    - cnf-lint --version
- usage in comand line
    - cfn-lint -t template1.yaml template2.yaml
- vs code extension
    - cloudFOrmation by kddejong
- if not then can be directly used in terminal using
    - cfn-lint <filename>
- example

        ```txt
        E1001 Top level template section Descrition is not valid
        lint-problems.yml:3:1

        E2002 Parameter SecurityGroup has invalid type AWS::EC2::SecurityGroup::I
        lint-problems.yml:10:5

        E3006 Resource My_instance has invalid name.  Name has to be alphanumeric.
        lint-problems.yml:22:3

        W1020 Fn::Sub isn't needed because there are no variables at Resources/My_instance/Properties/UserData/Fn::Base64/Fn::Sub
        lint-problems.yml:26:9

        E3008 Property "SecurityGroupIds" can Ref to parameter of types [String, AWS::SSM::Parameter::Value<String>, AWS::EC2::SecurityGroup::Id, AWS::SSM::Parameter::Value<AWS::EC2::SecurityGroup::Id>] at Resources/My_instance/Properties/SecurityGroupIds/0/Ref
        lint-problems.yml:34:11

        ```
#### 2
- lint can help with errors but writting the yaml is difficult as there are a lot of bolierplate syntax
- vs code extension : cloud formation snippets (also in codium)
- after installtion test with snippet_test.yml
    - open and type cfn then
        - ctrl space and select cfn
    - This will give a template
    - but this is too much so select cfn-lite instead
- description 
    - can be any text for info
- parameters
    - type param and ctrl space
    - select the first one to get a new template inserted for this section
    - remove paramters and replace it with InstanceType
    - description is just text
    - set default to t3.nano
- Resources
    - type ec2-in and ctrl space to get a lot of info
    - there is just too much here 
    - this is where the extension stops being helpful and
    - docs are to be approched to get info on whats required and whats optional
    - add ec2-secu ctrl space to get ec2-security group
    - search for ec2 security cloudformation and go to the docs
        - required is somethign to look in the docs
        - look for realworld exmaples in the end of the doc
- output
    - type output and ctrl space
    - 

### Resource
- [github cnf-lint](https://github.com/aws-cloudformation/cfn-lint)

### misc
 
---

## [beyond basics]()
Release date : 
### Idea
- 

### Details
#### 1 parameter types
- file : all prarmeter types.yml
- Upload it in the create stacks
- This will show all the paramters and its type with descp

#### 2 bulit in function
- to create references for resources both inside and outside the stack
- substituion and value refrences
- Two ways to invoke a function
    - Fn::Base64: some_string
    - !Base64 some_string
        - this method will not allow a function within a function
        - ex !Base64 !Ref my_param_name # error
        - !Base 64 Fn::Sub my_param_name # correct
- Substitution functions : used to fetch and use values from templates
    - Ref uses: 
        1. fetch param values
        2. make refernce to resources inside the templ (usually identifiers)
        
        ```yml
        Parameters:
            BucketName:
                Type: String
                Description: The name of a bucket.  This appears near the user prompt.
                Default: MyDefaultBucketName'
        # calling values using ref
        Bucket: !Ref BucketName

        # calling ref without !
        Bucket:
            Ref : BucketName
        ```
        - Ref behaviour
            - Usually gets ID of the resource
            - but its not consistent
            - so another function is used to make it consistent
    - GetAtt function
        - Fn::GetAtt:[resourceName, attribute]
        - !GetAttr resourceName.attribute
    - When to use ref
        - use ref for referring to user-entered param values and ID
        - for anythign else use GetAtt
    - sub function
        - Fn:Sub:
        - !Sub
    - 
    

### Resource
- [ref vs getattr](https://theburningmonk.com/cloudformation-ref-and-getatt-cheatsheet/)

### misc
 
---

## [advanced action]()
Release date : 
### Idea
- 

### Details
- 

### Resource
- 

### misc
 
---

## [composable and reuseable  templates]()
Release date : 
### Idea
- 

### Details
- 

### Resource
- 

### misc
 
---

## [automation]()
Release date : 
### Idea
- 

### Details
- 

### Resource
- 

### misc
 
---

## [intro]()
Release date : 
### Idea
- 

### Details
- 

### Resource
- 

### misc
 
---

