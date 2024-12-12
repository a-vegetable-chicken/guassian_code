"use strict";(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["ui_packages_ui-commands_ui-commands_ts"],{12480:(e,t,i)=>{i.d(t,{U0:()=>o});var n=i(97156);let s={Android:"Android",iOS:"iOS",macOS:"macOS",Windows:"Windows",Linux:"Linux",Unknown:"Unknown"};function o(){return function(){let e=s.Unknown,t=!1;if(n.cg){let i=n.cg.navigator,o=i.userAgent,a=i?.userAgentData?.platform||i.platform;-1!==["Macintosh","MacIntel","MacPPC","Mac68K","macOS"].indexOf(a)?e=s.macOS:-1!==["iPhone","iPad","iPod"].indexOf(a)?e=s.iOS:-1!==["Win32","Win64","Windows","WinCE"].indexOf(a)?e=s.Windows:/Android/.test(o)?e=s.Android:/Linux/.test(a)&&(e=s.Linux),t=i?.userAgentData?.mobile??(e===s.Android||e===s.iOS)}return{os:e,isAndroid:e===s.Android,isIOS:e===s.iOS,isMacOS:e===s.macOS,isWindows:e===s.Windows,isLinux:e===s.Linux,isDesktop:e===s.macOS||e===s.Windows||e===s.Linux,isMobile:t}}().isMacOS}},18558:(e,t,i)=>{i.d(t,{JC:()=>n.JC,KK:()=>n.KK,SK:()=>o,Vy:()=>n.Vy,ai:()=>n.ai,oc:()=>n.oc,rd:()=>n.rd});var n=i(50515);let s=/(?:^|,)((?:[^,]|,(?=\+| |$))*(?:,(?=,))?)/g;function o(e){return Array.from(e.matchAll(s)).map(([,e])=>e)}},98164:(e,t,i)=>{i.d(t,{$$:()=>r,GI:()=>o,zw:()=>s});var n=i(18558);let s=()=>{if("undefined"==typeof document)return!1;let e=document.querySelector("meta[name=keyboard-shortcuts-preference]");return!e||"all"===e.content},o=e=>/Enter|Arrow|Escape|Meta|Control|Mod|Esc/.test(e)||e.includes("Alt")&&e.includes("Shift"),a=new Set(["button","checkbox","color","file","hidden","image","radio","range","reset","submit"]),r=e=>{let t=(0,n.Vy)(e),i=s()&&!function(e){if(!(e instanceof HTMLElement))return!1;let t=e.nodeName.toLowerCase(),i=e.getAttribute("type")?.toLowerCase()??"text",n="true"===e.ariaReadOnly||"true"===e.getAttribute("aria-readonly")||null!==e.getAttribute("readonly");return("select"===t||"textarea"===t||"input"===t&&!a.has(i)||e.isContentEditable)&&!n}(e.target);return o(t)||i}},35311:(e,t,i)=>{i.d(t,{J:()=>s,k:()=>CommandEvent});var n=i(89794);let CommandEvent=class CommandEvent{constructor(e){this.commandId=e}};let s={entries:e=>Object.entries(e).filter(e=>n.dx.is(e[0])&&void 0!==e[1]),keys:e=>Object.keys(e).filter(n.dx.is)}},58033:(e,t,i)=>{i.d(t,{J:()=>d,c:()=>r});var n=i(96540),s=i(35311),o=i(89794);let a=new Map;function r(){let e=new Map;for(let t of new Set(Array.from(a.values()).flat())){let i=o.dx.getServiceId(t);if(!e.has(i)){let t=(0,o.tp)(i);e.set(i,{service:{id:t.id,name:t.name},commands:[]})}let n=(0,o.fL)(t);n&&n.defaultBinding&&e.get(i)?.commands.push({id:t,name:n.name,description:n.description,keybinding:n.defaultBinding})}return Array.from(e.values())}let d=e=>{let t=(0,n.useId)();(0,n.useEffect)(()=>(a.set(t,s.J.keys(e)),()=>{a.delete(t)}),[e,t])}},89794:(e,t,i)=>{i.d(t,{dx:()=>d,fL:()=>l,xJ:()=>c,eY:()=>m,tp:()=>u});var n=i(97564),s=i(18558);let{P:o,$:a}=JSON.parse('{"$":{"commit-diff-view":{"id":"commit-diff-view","name":"Commits diff view","commandIds":["commit-diff-view:open-find","commit-diff-view:create-permalink","commit-diff-view:go-up-commit","commit-diff-view:go-up-commit-arrow","commit-diff-view:go-down-commit","commit-diff-view:go-down-commit-arrow","commit-diff-view:tab-forward","commit-diff-view:tab-backward"]},"copilot-chat":{"id":"copilot-chat","name":"Copilot Chat","commandIds":["copilot-chat:send-message","copilot-chat:send-message-new-conversation","copilot-chat:add-attachment","copilot-chat:stop-response"]},"github":{"id":"github","name":"GitHub (site-wide)","commandIds":["github:submit-form","github:select-multiple"]},"issue-create":{"id":"issue-create","name":"Issue Create","commandIds":["issue-create:new","issue-create:submit-and-create-more","issue-create:open-fullscreen"]},"issue-viewer":{"id":"issue-viewer","name":"Issue viewer","commandIds":["issue-viewer:edit-parent","issue-viewer:edit-title-submit","issue-viewer:close-edit-title"]},"issues-react":{"id":"issues-react","name":"Issues","commandIds":["issues-react:focus-next-issue","issues-react:focus-previous-issue"]},"item-pickers":{"id":"item-pickers","name":"Item Pickers","commandIds":["item-pickers:open-assignees","item-pickers:open-development","item-pickers:open-labels","item-pickers:open-milestone","item-pickers:open-projects","item-pickers:open-issue-type","item-pickers:open-author"]},"keyboard-shortcuts-dialog":{"id":"keyboard-shortcuts-dialog","name":"Keyboard Shortcuts Dialog","commandIds":["keyboard-shortcuts-dialog:show-dialog"]},"list-view-items-issues-prs":{"id":"list-view-items-issues-prs","name":"List View Items: Issues & Pull Requests","commandIds":["list-view-items-issues-prs:open-focused-item","list-view-items-issues-prs:toggle-focused-item-selection"]},"list-views":{"id":"list-views","name":"List views including lists of issues, pull requests, discussions, and notifications.","commandIds":["list-views:open-manage-item-dialog"]},"projects":{"id":"projects","name":"Projects","commandIds":["projects:save-view","projects:save-form"]},"pull-requests-diff-file-tree":{"id":"pull-requests-diff-file-tree","name":"Pull requests - diff file tree","commandIds":["pull-requests-diff-file-tree:focus-file-tree"]},"pull-requests-diff-view":{"id":"pull-requests-diff-view","name":"Pull requests - \'Files changed\' view","commandIds":["pull-requests-diff-view:copy-code","pull-requests-diff-view:expand-all-hunks","pull-requests-diff-view:expand-hunk-up","pull-requests-diff-view:expand-hunk-down","pull-requests-diff-view:copy-anchor-link","pull-requests-diff-view:start-conversation-current","pull-requests-diff-view:jump-to-next-result","pull-requests-diff-view:jump-to-next-result-alternate","pull-requests-diff-view:jump-to-previous-result","pull-requests-diff-view:jump-to-previous-result-alternate","pull-requests-diff-view:open-find","pull-requests-diff-view:close-find"]},"react-sandbox":{"id":"react-sandbox","name":"React Sandbox","commandIds":["react-sandbox:example-command"]},"sub-issues":{"id":"sub-issues","name":"sub-issues","commandIds":["sub-issues:create-sub-issue","sub-issues:add-existing-issue"]}},"P":{"commit-diff-view:create-permalink":{"name":"Create permalink","description":"Hotkey to expand the current url to a full permalink.","defaultBinding":"y"},"commit-diff-view:go-down-commit":{"name":"Go up commit","description":"navigates up one commit","defaultBinding":"j"},"commit-diff-view:go-down-commit-arrow":{"name":"Go up commit","description":"navigates down one commit","defaultBinding":"ArrowDown"},"commit-diff-view:go-up-commit":{"name":"Go up commit","description":"navigates up one commit","defaultBinding":"k"},"commit-diff-view:go-up-commit-arrow":{"name":"Go up commit","description":"navigates up one commit","defaultBinding":"ArrowUp"},"commit-diff-view:open-find":{"name":"Open up find and search on selection","description":"Hotkey to open up the custom find and search on selection.","defaultBinding":"Mod+e"},"commit-diff-view:tab-backward":{"name":"Track when a user tabs backwards through the commit list","description":"navigates through the tab-index items in the commit list backwards","defaultBinding":"Shift+Tab"},"commit-diff-view:tab-forward":{"name":"Track when a user tabs through the commit list","description":"navigates through the tab-index items in the commit list","defaultBinding":"Tab"},"copilot-chat:add-attachment":{"name":"Add attachment","description":"Add an attachment to the current message","defaultBinding":"Mod+Shift+@"},"copilot-chat:send-message":{"name":"Send message","description":"Send the current message","defaultBinding":"Enter"},"copilot-chat:send-message-new-conversation":{"name":"Send message to new conversation","description":"Send the current message to a newly created conversation","defaultBinding":"Mod+Shift+S"},"copilot-chat:stop-response":{"name":"Stop response","description":"Stop streaming the current response","defaultBinding":"Escape"},"github:select-multiple":{"name":"Select multiple items","description":"Add the current item to a multi-selection (or remove it if already added)","defaultBinding":"Mod+Enter"},"github:submit-form":{"name":"Submit form","description":"Submit the current form","defaultBinding":"Mod+Enter"},"issue-create:new":{"name":"Create a new issue","description":"Initiate new issue creation","defaultBinding":"c"},"issue-create:open-fullscreen":{"name":"Open issue creation in fullscreen","description":"Open the issue creation dialog in fullscreen mode","defaultBinding":"Mod+Shift+Enter"},"issue-create:submit-and-create-more":{"name":"Submit and create more","description":"Submit the current issue and create a new one","defaultBinding":"Mod+Alt+Enter"},"issue-viewer:close-edit-title":{"name":"Cancel","description":"Cancel out of editing an issue\'s title","defaultBinding":"Escape"},"issue-viewer:edit-parent":{"name":"Edit parent","description":"Edit parent for current issue","defaultBinding":"Alt+Shift+P"},"issue-viewer:edit-title-submit":{"name":"Save","description":"Submit changes made to an issue\'s title","defaultBinding":"Enter"},"issues-react:focus-next-issue":{"name":"Focus on Next Issue","description":"Focus on the next issue in the list, or the first one if none are focused.","defaultBinding":"j"},"issues-react:focus-previous-issue":{"name":"Focus on Previous Issue","description":"Focus on the previous issue in the list","defaultBinding":"k"},"item-pickers:open-assignees":{"name":"Open assignees panel","description":"Open panel to select assignees","defaultBinding":"a"},"item-pickers:open-author":{"name":"Open author panel","description":"Open panel to select author","defaultBinding":"u"},"item-pickers:open-development":{"name":"Open development panel","description":"Open panel to create or link a pull request","defaultBinding":"d"},"item-pickers:open-issue-type":{"name":"Open issue type panel","description":"Open panel to select issue type","defaultBinding":"t"},"item-pickers:open-labels":{"name":"Open labels panel","description":"Open panel to select labels","defaultBinding":"l"},"item-pickers:open-milestone":{"name":"Open milestone panel","description":"Open panel to select milestone","defaultBinding":"m"},"item-pickers:open-projects":{"name":"Open projects panel","description":"Open panel to select projects","defaultBinding":"p"},"keyboard-shortcuts-dialog:show-dialog":{"name":"Show Keyboard Shortcuts Dialog","description":"Display the keyboard shortcuts help dialog","defaultBinding":"Shift+?"},"list-view-items-issues-prs:open-focused-item":{"name":"Open Focused Item","description":"Open the currently focused item","defaultBinding":"o"},"list-view-items-issues-prs:toggle-focused-item-selection":{"name":"Toggle Focused Item Selection","description":"Toggle the selection state of the currently focused item for bulk actions","defaultBinding":"x"},"list-views:open-manage-item-dialog":{"name":"Open \'manage item\' dialog","defaultBinding":"Mod+Shift+U","description":"Open a dialog to manage the currently focused item."},"projects:save-form":{"name":"Save","description":"Submits the currently focused form.","defaultBinding":"Mod+Enter"},"projects:save-view":{"name":"Save view","description":"Save any unsaved changes to the current project view.","defaultBinding":"Mod+s"},"pull-requests-diff-file-tree:focus-file-tree":{"name":"Focus file tree","description":"Move focus to the file tree","defaultBinding":"Mod+F6"},"pull-requests-diff-view:close-find":{"name":"Close Find","description":"Close the find window","defaultBinding":"Escape"},"pull-requests-diff-view:copy-anchor-link":{"name":"Copy link","description":"Copy link to the current line","defaultBinding":"Mod+Alt+y"},"pull-requests-diff-view:copy-code":{"name":"Copy","description":"Copy the code for the current line(s)","defaultBinding":"Mod+c"},"pull-requests-diff-view:expand-all-hunks":{"name":"Expand all","description":"Expand all hunks in the current file","defaultBinding":"Mod+Shift+ArrowLeft"},"pull-requests-diff-view:expand-hunk-down":{"name":"Expand below","description":"Expand the current hunk downward"},"pull-requests-diff-view:expand-hunk-up":{"name":"Expand above","description":"Expand the current hunk upward"},"pull-requests-diff-view:jump-to-next-result":{"name":"Jump to the next search result","description":"Jump to the next search result","defaultBinding":"Enter"},"pull-requests-diff-view:jump-to-next-result-alternate":{"name":"Jump to the next search result","description":"Jump to the next search result","defaultBinding":"Mod+g"},"pull-requests-diff-view:jump-to-previous-result":{"name":"Jump to the previous search result","description":"Jump to the previous search result","defaultBinding":"Shift+Enter"},"pull-requests-diff-view:jump-to-previous-result-alternate":{"name":"Jump to the previous search result","description":"Jump to the previous search result","defaultBinding":"Mod+Shift+G"},"pull-requests-diff-view:open-find":{"name":"Open up find","description":"Hotkey to open up the custom find.","defaultBinding":"Mod+f"},"pull-requests-diff-view:start-conversation-current":{"name":"Start conversation on line","description":"Start a conversation on the current line"},"react-sandbox:example-command":{"name":"React Sandbox Example Command","description":"Do something.","defaultBinding":"Mod+Shift+Enter"},"sub-issues:add-existing-issue":{"name":"Add existing issue","description":"Add an existing issue as a sub-issue","defaultBinding":"Alt+Shift+A"},"sub-issues:create-sub-issue":{"name":"Create sub-issue","description":"Create a new sub-issue","defaultBinding":"Alt+Shift+C"}}}'),r=new Set(Object.keys(o)),d={is:e=>r.has(e),getServiceId:e=>e.split(":")[0]},l=e=>{let t=o[e];return!t.featureFlag||(0,n.G7)(t.featureFlag.toUpperCase())?t:void 0},u=e=>a[e],c=e=>{let t=l(e);return t?.defaultBinding?(0,s.rd)(t.defaultBinding):void 0},m=e=>new Map(e.map(e=>[e,c(e)]).filter(e=>void 0!==e[1]))},26750:(e,t,i)=>{i.d(t,{Vr:()=>C,cQ:()=>I,ky:()=>s.k,N5:()=>O,hh:()=>E,ak:()=>b,tL:()=>L});var n,s=i(35311),o=i(89794),a=i(74848),r=i(80497),d=i(96540),l=i(58033);let u=new(i(18679)).s({collectorUrl:"https://collector.githubapp.com/ui-commands/collect"}),c={TYPE:"command.trigger",send(e){u.sendEvent(c.TYPE,e)}};function m(e,t){c.send({command_id:e.commandId,trigger_type:t instanceof KeyboardEvent?"keybinding":"click",target_element_html:t.target instanceof HTMLElement?function(e){let t=e.tagName.toLowerCase(),i=Array.from(e.attributes).map(e=>`${e.name}="${e.value.replaceAll('"','\\"')}"`).join(" ");return`<${t}${i?` ${i}`:""}>`}(t.target):void 0,keybinding:(0,o.xJ)(e.commandId)})}let p=new Map;function f(e,t){let i=(0,d.useMemo)(()=>new Map,[]),n="global"===e?p:i;(0,d.useEffect)(()=>{for(let[e,i]of(0,o.eY)(s.J.keys(t))){let t=n.get(i)?.filter(t=>t!==e)??[];t.length&&console.warn(`The keybinding (${i}) for the "${e}" command conflicts with the keybinding for the already-registered command(s) "${t.join(", ")}". This may result in unpredictable behavior.`),n.set(i,t.concat(e))}return()=>{for(let[e,i]of(0,o.eY)(s.J.keys(t))){let t=function(e,t){let i=!1;return e.filter(e=>e!==t||!!i||(i=!0,!1))}(n.get(i)??[],e);t?.length?n.set(i,t):n.delete(i)}}},[t,n])}var g=i(18558),v=i(98164);function h(e,t,i){let n=(0,d.useMemo)(()=>new g.KK,[]),s=(0,d.useMemo)(()=>{let t=new Map;for(let i of e){let e=(0,o.xJ)(i);e&&t.set(e,i)}return t},[e]),a=(0,d.useRef)(null);return(0,d.useCallback)(e=>{let o="nativeEvent"in e?e.nativeEvent:e;if(a.current===o)return;if(a.current=o,!(0,v.$$)(o)){n.reset();return}n.registerKeypress(o);let r=s.get(n.sequence)??s.get((0,g.Vy)(o));r&&(void 0===i||i())&&(n.reset(),e.preventDefault(),e.stopPropagation(),o.stopImmediatePropagation(),t(r,o))},[s,n,t,i])}let w="ui-command-trigger",b=({commands:e})=>{let t=(0,d.useRef)(null),i=(0,d.useCallback)((t,i)=>{let n=e[t];if(n){let e=new s.k(t);try{n(e)}finally{m(e,i)}}},[e]),n=(0,d.useCallback)(()=>{let e=function(){let e=[...document.querySelectorAll('dialog:modal, [role="dialog"][aria-modal="true"]')].filter(e=>e.childNodes.length>0&&function e(t){if(t.clientHeight>0)return!0;for(let i of t.children)if(e(i))return!0;return!1}(e));return e.length?e[e.length-1]:null}();return!e||function(e,t){return!!t&&(e.contains(t)??!1)}(e,t.current)},[]),r=h(s.J.keys(e),i,n);return f("global",e),(0,l.J)(e),(0,d.useEffect)(()=>{let e=e=>{let t="detail"in e&&"object"==typeof e.detail?e.detail:void 0;if(!t)return;let n="commandId"in t&&"string"==typeof t.commandId&&o.dx.is(t.commandId)?t.commandId:void 0,s="domEvent"in t&&(t.domEvent instanceof KeyboardEvent||t.domEvent instanceof MouseEvent)?t.domEvent:void 0;n&&s&&i(n,s)};return document.addEventListener("keydown",r),document.addEventListener(w,e),()=>{document.removeEventListener("keydown",r),document.removeEventListener(w,e)}},[r,i,t]),(0,a.jsx)("div",{ref:t,className:"d-none"})};try{b.displayName||(b.displayName="GlobalCommands")}catch{}let y=(0,d.createContext)({triggerCommand:function(e,t){document.dispatchEvent(new CustomEvent(w,{detail:{commandId:e,domEvent:t}}))}}),k=y.Provider,x=()=>(0,d.useContext)(y);var S=i(30871);let E=({commandId:e,...t})=>{let i=(0,o.xJ)(e);return i?(0,a.jsx)(S.U,{keys:i,...t}):null};try{E.displayName||(E.displayName="CommandKeybindingHint")}catch{}let C=(0,d.forwardRef)(({commandId:e,children:t,description:i,leadingVisual:n,trailingVisual:s,...d},l)=>{let u=(0,o.fL)(e),{triggerCommand:c}=x();return u?(0,a.jsxs)(r.l.Item,{...d,onSelect:t=>c(e,t.nativeEvent),ref:l,children:[t??u.name,i?(0,a.jsx)(r.l.Description,{truncate:!0,children:i}):null,n?(0,a.jsx)(r.l.LeadingVisual,{children:n}):null,null!==s&&(0,a.jsx)(r.l.TrailingVisual,{children:s??(0,a.jsx)(E,{commandId:e,format:"condensed"})})]}):null});C.displayName="ActionList.CommandItem";var B=i(55847);let I=(0,d.forwardRef)(({commandId:e,children:t,trailingVisual:i,showKeybindingHint:n=!1,keybindingHintVariant:s,...r},d)=>{let l=(0,o.fL)(e),{triggerCommand:u}=x();if(!l)return null;let c=s??("primary"===r.variant?"onEmphasis":"normal");return(0,a.jsx)(B.Q,{...r,onClick:t=>u(e,t.nativeEvent),trailingVisual:i??n?()=>(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)("span",{className:"sr-only",children:"("}),(0,a.jsx)(E,{commandId:e,format:"condensed",variant:c}),(0,a.jsx)("span",{className:"sr-only",children:")"})]}):void 0,ref:d,children:t??l.name})});I.displayName="CommandButton";try{(n=HintVisual).displayName||(n.displayName="HintVisual")}catch{}var M=i(87330);let O=(0,d.forwardRef)(({commandId:e,"aria-label":t,...i},n)=>{let s=(0,o.fL)(e),{triggerCommand:r}=x();return s?(0,a.jsx)(M.K,{"aria-label":t??s.name,onClick:t=>r(e,t.nativeEvent),ref:n,keybindingHint:(0,o.xJ)(e),...i}):null});O.displayName="CommandIconButton";var j=i(98152),q=i(10120),A=i(64515);let L=(0,d.forwardRef)(({commands:e,as:t,...i},n)=>{let o=(0,q.M)(e),r=x(),u=(0,d.useCallback)((e,t)=>{let i=o.current[e];if(i){let n=new s.k(e);try{i(n)}finally{m(n,t)}}else r.triggerCommand(e,t)},[o,r]);f("scoped",e),(0,l.J)(e);let c=(0,d.useMemo)(()=>({triggerCommand:u}),[u]),p=h(s.J.keys(e),u),g=(0,j._)(p),v=(0,d.useRef)(null);(0,A.T)(n,v),(0,d.useEffect)(()=>{let e=v.current,t=g.onKeyDown;if(e)return e.addEventListener("keydown",t),()=>e.removeEventListener("keydown",t)});let w=void 0!==t||void 0!==i.className?void 0:{display:"contents"};return(0,a.jsx)(k,{value:c,children:(0,a.jsx)(t??"div",{style:w,...i,...g,ref:v})})});L.displayName="ScopedCommands"},98152:(e,t,i)=>{i.d(t,{_:()=>a});var n=i(12480),s=i(96540);let o=new Set(["enter","tab"]),a=e=>{let t=(0,s.useRef)(!1),i=(0,s.useRef)(!1),a=(0,s.useCallback)(e=>{"compositionstart"===e.type&&(t.current=!0,i.current=!1),"compositionend"===e.type&&(t.current=!1,i.current=!0)},[]),r=(0,s.useCallback)(s=>{if(!o.has(s.key.toLowerCase())||!t.current){if((0,n.U0)()&&229===s.keyCode&&i.current){i.current=!1;return}e(s)}},[e]);return(0,s.useMemo)(()=>({onCompositionStart:a,onCompositionEnd:a,onKeyDown:r}),[a,r])}},10120:(e,t,i)=>{i.d(t,{M:()=>o});var n=i(51012),s=i(96540);function o(e){let t=(0,s.useRef)(e);return(0,n.N)(()=>{t.current=e},[e]),t}}}]);
//# sourceMappingURL=ui_packages_ui-commands_ui-commands_ts-0ed281a96755.js.map