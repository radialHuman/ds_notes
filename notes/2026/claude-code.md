(cheatsheet)[https://awesomeclaude.ai/code-cheatsheet]

## what problem does it solve
1. makes coding way faster
2. but was the the issue, debateable

## Different from
1. claude.ai : on browser on on your system
2. claude desktop : on your system, connects to sharepoint and other things as your proxy; can control your system (like manus)
    - might slow your system
3. claude code : opens in terminal or ide, mainly for tech but can be used by non tech
4. claude plugins : ppts, excels, docs, like copilot

## not just an llm
- it has connection to a lot of tools
- decided when to call
- how many times to call
- has memory
- can plan and execute without you

## Not only for technical folks
- automate things on your system with just prompting
- manipulate files by prompting
- create your won temporary software to do something
- schedule a/many tasks
- loop 

## Rules
1. always know what you want, else there is no way to verify what it has done
2. be in control, else it will take you for a ride
3. if you dont know, start in plan mode
4. ask it questions, understand and gain knowledge, then start
5. there is always more to know, even if you are an sme
6. ask it to check for issue when done
7. learn a bit about markdown files
    1. \---
    2. \#
8. speed : Haiku > sonnet > opus
9. cost and correctness : Haiku < sonnet < opus


## Cheat sheet
1. open in code editor to see .claude settings and changes in the folder
1. /help : has multiple options to toggle and see options
    1. /commands : commands list
    2. make custom commands
    3. /model : select which model to use
    4. /tasks : in case multiple cc sesisons are running, it will show
        - ctrl + t : same thing
        - k : kill the task
        - it can be like running a webserver in the background (since tis a blocking task)
        - esc : to exit
    5. /exit
1. ! : bash mosde
1. / : for various commands
1. @ : reference a file
1. & : background running 
1. ctrl+o : show verbose, what is it thinking?
1. ctrl+b : run the session in background
1. ctrl + sht + - : undo
1. Shft + tab : changes modes : coding , planning, asking
1. double esc : clear what ever you have written
1. up down : previous and next prompts
1. alt/option + enter : new line in prompt
1. toggle up and down to select which option to choose when it asks questions/permissions
    1. it will, if applicable, will give option to type in your custom option too
1. initiate git if working on code using cc
    1. just say to always commit before changging so that a working version is always available
1. take screenshot of whats not loooking good, or something that can be used as a inspiration 
    - save ti in the code base and call it via @ and add prompt

## memory
1. you run a session ask to always git things
    - after the work is done you close the terminal
    - what happens next time? it wont follow the instruction
2. Claude.md 
    - can contain informaiton about the project. like a README but for claude
    - persistant memory
    - /init : to create the file or create it yourself
    - to update : type what you want to update in it and mention "in @claude.md"


## agents
1. /agents : multiple claude bots running and doing things (costly)
1. complicated stuff, like having multiple interns working on a big codebase

## MCP
1. /mcp : shows which 3rd party servics you can connect to in order to automate some task
1. happens only if 3rd party has provided mcp servers to interact with and we have permissions
2. ex : jira, snow, drive etc.

## Skills
1. skills.md
    - general instructions written once, in the same folder, used everytime claude reads that
    - like if you wan it to address you as the supreme leader everytime
    - of have a watermark in all your work
    - can use someone elses via plugin but BE VERY CAREFUL (new attack surface)
2. have multiple of them for various tasks
    - reference them in your prompt to nudge cc to use it
3. can have it inside .claude/skills/[multiple skills file here]



## Issues
1. time out 500 error after ~18 mins (talk to the middle man)
2. too much thinking and tokens burning (environmental damage, check out protests against datacenters)
3. needs proper control and guidance, else it will do what it knows not what you want
4. dont use ferrari to buy milk
5. dont use nano on a race track
6. throwaway code
7. cognitive offloading
    - once in a while code in notepad to mantain plasticity
    - the day it will not work (anthropic downtime), dont become useless
8. Narrows options
    - its trained on python and js a lot
    - that too particular libraries
    - that too a version of that library
    - does it fetch the doc when things changes?
9. Single source of knowledge
    - google provided options to choose from (though its bad now)
    - this one is not even giving us that
    - example : image color
10. it will do what its trained on, it was trained 6 months back
    - does it know the new stuff?
    - will it recommend that?
    - wont that affect your quality?
11. Do you trust yourself?
    - do you know the entire context?
    - do you understand the ask?
    - do you know what you dont know?
12. ram:hdd::you:llms