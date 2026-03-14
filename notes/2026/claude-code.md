## Poll
1. how many of you, in your work feel you are at global level?
    - your work and your expertise is at the level of experts out there

## what problem does it solve
1. makes coding way faster
    - its a ability booster
    - higher your ability, higher its gains
    - e=?
2. but was that the issue, debateable
    - tradeoffs (discussed in the end)
- for business : its as simple as this
    - write the word and it will be done (you can now leave :P)
    - it makes making ppts, excel analysis, wriritng emails and other mundane tasks faster and "professional"
3. not the only one who is solving, we are using because we have a deal
---
## Different flavours
1. claude.ai : on browser on on your system
    - connectors 
    - skill hub
2. claude desktop : claude.ai on your system, connects to sharepoint and other things as your proxy; can control your system (like manus)
    - might slow your system
    - connectors
    - skill hub
3. claude code : opens in terminal or ide, mainly for tech but can be used by non tech
    - not to be confused by the general terminal interface, this one only seems like you are a hacker
        - but you just need to know english
        - and what you are doing (thats the catch)
    - connectors
    - skill hub
    - mcp
4. claude plugins : ppts, excels, docs, like copilot
---
## not just an llm, understand the terminology
- claude : bunch of llms provided by anthropic. Using it since 2024
    - sonnet, haiku and opus
    - similar to 
        - gpt 4, gpt 5 or gemini 2.5 or grok or llama 3.1 or deepseek 2
- so when you say use claude, what do you mean?
    - use the llm in my application?
    - use llm to generate code?
    - use claude.ai, claude desktop, claude code, coworker?
    - its like asking use microsoft for your next project
        - word? excel? notepad? xbox?
        - what if I am running linux?
    - instead say 
        - use claudes various versions as your prefered llm in the applciations **if possible**
        - use claude in vscode as it seems to have expertise in coding (not really, for us its almost the same)
        - use claude-code to accelerate as the timeline has moved forward
        - use claude-desktop to automate this task on your system, if you dont know how to do it in a simpler manner
        - use claude-plugins to finish this ppt, if you are lazy
        - use claude-x because we have purchased it to raise our SP and we dont care about the dire ramifications (all s/w has it, more details towards the end)

- llm is a text in text out system (images in some and audio and video an others)
    - it doesn execute your code
    - it doesn connect to anything
    - you need connectors (mcp is just one of them)
- it has connection to a lot of tools
    - it will do almost everything thats required in day to day work
    - decided when to call
    - how many times to call
    - has memory
    - can plan and execute without you
- two different things : llm and its ecosystem of tool/functions/pieces of code and connectors
---
## Not only for technical folks
- automate things on your system with just prompting
- manipulate files by prompting
- create your won temporary software to do something
- schedule a/many tasks
    - cron jobs
- /loop 
    - it will keep n doing till the goal is achieved
    - max 3 days in the same session
---
## Rules
0. got stuck? ask it
1. always know what you want, else there is no way to verify what it has done
2. be in control, else it will take you for a ride
3. if you dont know, start in plan mode
    - dont code before finalizing the architecture
    - ask for recommendations, options, comparisions
    - for that you need to know what you are doing in and out
    - always ask WWRS? find flaws and loopholes
4. ask it questions, understand and gain knowledge, then start
5. there is always more to know, even if you are an sme
6. ask it to check for issue when done
7. learn a bit about markdown files
    1. \---
    2. \#
    3. \**
8. speed : Haiku > sonnet > opus
9. cost and correctness : Haiku < sonnet < opus
    - 1 token = ~3/4 word (both input and output)
10. Maintain the context, dont bloat your skills.md and CLAUDE.md
    - dont think of this as dump and ask chatbot
    - irrespective of context window size, context rot will occur after a point
---

## Cheat sheet
0. extended thinking is on by default, so turn it off for speed-accuracy tradeoff
1. open in code editor to see .claude settings and changes in the folder
1. /help : has multiple options to toggle and see options
    1. /commands : commands list
    2. make custom commands, not manually
    3. /model : select which model to use
    4. /tasks : in case multiple cc sesisons are running, it will show
        - ctrl + t : same thing
        - k : kill the task
        - it can be like running a webserver in the background (since tis a blocking task)
        - esc : to exit
    5. /exit
    7. ctrl+c 2x : exit
    6. /clear : clears memory
    7. /compact : to compress memory of the session and keep just the keep info when it reaches a threshold
        - /compress  keep api info
        - but you should know whats important ot keep and whats not
        - else rely on it and get burnt
    8. Flags : whhen starting a session
        - claude --resume : to start the previous sesssion
        - --snonet : use sonnet
        - --allowedTools
        - --verbose
        - --dangerously-skip-permissions
        - --worktree / -w followed by directory/branch name
    9. /stats : tokens used
    10. /cost : cost incurred
    11. /context : to look at context window status
    12. /insights : creates pdf analysisng the code base and reports
    13. /permissions : to check whats allowed and denied
    14. /chrome : browser control mode. can debug erro in website
    0. create your own commands by putting them in .claude/commands folder
1. ! : bash mosde
1. / : for various commands
1. @ : reference a file
1. & : background running 
1. ctrl+o : show verbose, what is it thinking?
1. ctrl+b : run the session in background
1. ctrl + sht + - : undo
1. Shft + tab : changes modes : coding , planning, asking
1. esc : if it goes down a rabbit hole. interupt and redirect
1. double esc : clear what ever you have written
1. up down : previous and next prompts
1. alt/option + enter : new line in prompt
1. toggle up and down to select which option to choose when it asks questions/permissions
    1. it will, if applicable, will give option to type in your custom option too
1. initiate git if working on code using cc
    1. just say to always commit before changging so that a working version is always available
1. Multi-modeal : take screenshot of whats not looking good, or something that can be used as a inspiration 
    - save ti in the code base and call it via @ and add prompt
1. checkpoints : snap shot after every change to backtrack, because it knows it can go down the wrong path
    - /rewind : to go back to a point in  time when things were correct
1. headless mode/cli mode 
    - [prompt] -p
1. Ralph loop : dangerous
1. worktree : like dev qa and prod. 
    - individual features and respective branches to work on
    - fixes bug in one while not touching the others
    - can be combined with sub-agents
    - later merge them
1. ask claude to build a status bar for context area
1. drag and drop screenshots or images into terminal
---
## claude settings.json
- permissions : allow and dont allow to control what permissions it needs to ask and what it doesn
    - allowed example : git, cat, ls, etc (you need to know)
    - deny : delete, rm, mv, chmod etc (else you lose control)
        - reading .env or other senstive files
---
## .claude folder in project
- not to be confused with global .claude folder
- be careful, build it with care. dont dump
- keep files less than 500 lines and not multiple files
- version control it
- modify it if its not acting in a desriable manner
- /agent : workers running in background
- /hooks : things to run everytime somehting happens. like custom script to be executed instead of gettting it generated. To save tokens
    - event driven shell commands
    - has pre defined events and triggers
    - set it up in setting.json
    - exmaples
        - commit everytime
        - auto formatter
        - look for certain words and mask them
        - notification
    - /hookify
        - create own hooks
    - prehook , posthook
- /commands : custom shortcuts
- /skills : things or ability to follow 
- /rules : break down the CLAUDE.md files
    - code-stle.md
    - frontend
        - react.md
    - security.md
    - testing.md
- .mcp.json
- settings.json
- settings.local.json
- TASKS.md : todo
- PROGRESS.md : whats done
---
## memory
1. you run a session ask to always git things
    - after the work is done you close the terminal
    - what happens next time? it wont follow the instruction
    - /memory/MEMORY.md
2. CLAUDE.md  (new study shows that they have side effects and extra cost)
    - rulebook for the project
    - tech stack, HL architecture, design patterns, validation and build flow
    - top to bottom reduces importance
    - give exmaples
    - capture the uniqueness of the project
    - exmaple
        - dont write so many documentation files
        - address me as the "commander in chief" everytime you talk to me
        - follow this coding style and style guide always
    - can contain informaiton about the project. like a README but for claude
    - persistant memory
    - /init : to create the file or create it yourself
    - to update : type what you want to update in it and mention "in @claude.md"
        - dont do it manually
3. its not unlimited, it will only know so much based on your conversation
    - either it will compress
    - or it will slow down
    - context window needs to be considered
---
## agents
1. /agents : multiple claude bots running and doing things (costly)
1. name them like headlines
```markdown
---
name : 
description : 
tools : 
model :
---
body of the agent, descript what to do, how to do etc
```
1. complicated stuff, like having multiple interns working on a big codebase
1. sub-agents (specialized siloed workers)
    - separate context window
    - runs parallely
    - own system prompt
    - own tool access
    - but follows the main CLAUDE.md file
    - independent of each other
1. Main agent : coordinates between sub agents that are siloed
1. Agent Teams (costly, experimental)
    - works like a team with inter communication
    - has a shared task list
---
## MCP
1. /mcp : shows which 3rd party servics you can connect to in order to automate some task
    1. /plugin : can be found using this too
1. happens only if 3rd party has provided mcp servers to interact with and we have permissions
2. ex : jira, snow, drive, ms365 etc.
1. llm doesn connect to anything. mcp is the common way, but not the only one
---
## Skills
0. inside [skils-name] folder
1. skills.md
    - general instructions written once, in the same folder, used everytime claude reads that
    - like if you wan it to address you as the supreme leader everytime
    - of have a watermark in all your work
    - can use someone elses via plugin but BE VERY CAREFUL (new attack surface)
2. have multiple of them for various tasks
    - reference them in your prompt to nudge cc to use it
    - name them in a nice way so that its easy to refer to it in prompt
3. can have it inside .claude/skills/[multiple skills file here]
4. Use skills creator to match your taste
5. have too many and your context will overflow
6. format with front-matter and body
```markdown
--- 
name : 
description : 
---

Body of the skills
```
7. references
8. assets
9. scripts
---

## Plugins
1. bundle of skils, mcp, agent, hooks created for specific use
2. dont use random ones


## Sample folder
[github repo](https://github.com/ChrisWiles/claude-code-showcase/blob/main/.claude)

## Your role
- become a manager but the one who knows the traps and the way to get out of it
    - you need to question
    - it needs to answer 
    - you need to know if its bluffing or misguided
    - if you dont, then its all magic to you
    - then you are just a middle man, that will eb replaced. 
    - your moat is your expertise and understanding
0. Before writing the code
    - plan
    - get all the details from business
    - give all the details to claude code
    - ask for options and how tos
1. While writing the code
    - keep commiting
    - check the changes
    - check the logic
    - validate the process and plan 
2. After the work is ready
    - check for vulnerabilities
    - run it through checkmarxs
    - ask for whats missing?
    - what have we not thought of?
    - explain what you have done
    - why did you do that?
    - how does that work?
3. If using a non approved skill
    - dont
    - clean it up (if you know how to)
4. become a task reviewer
    - you think wirtting was easy
    - welcome to the age of reviewing
    - or just hit the skip button and blame the llm, when things hit the fan
---
## what if
1. you want to remove last x pages from all the pdfs in a location
2. you want to rename multiple files if it has a word in its name
3. you just came back from vacation and found the demo is this afternoon
4. before starting a project on something you haven worked on previously
5. you have been asked to give a training on a fancy new tool in company
6. need to prepare doc for a legacy codebase
7. boss wants a ppt on something
8. you want to use llms in your task in a proper manner
---
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
12. will increase attach surface
13. social media shows whats easy but not whats right
    - Openclaw aka Clawdbot aka Moltbot aka Molty is a security nightmare
    - use critical thinking and dont fall for the hype
13. adding others text/prompt (like npm libraries) will lead to security issues
12. ram:hdd::you:llms
    - Gear acquisition syndrome
        - There are a lot of plugins and tools to enhance your clude-code
            - but that doesn guarantee you performance
            - think of this as adding all the modifications to your car, mileage will not change a lot
            - mileage depends on the fuel quality, you are the fuel
    - focus on your upskilling not just updating our system with the latest tools
        - "It's not the car you drive.  It's the driver who's driving the car that's doing the driving." 
13. given the track record, convinenece will aways win, short term gain, long term loss
14. this will reduce the workforce eventually


(cheatsheet)[https://awesomeclaude.ai/code-cheatsheet]