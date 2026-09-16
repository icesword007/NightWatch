import os
import re
import shlex


MAX_TASK_INPUT_COMMAND_CHARS = 4_096
DEFAULT_MAX_DEPTH = 6
DEFAULT_MAX_ENTRIES = 4_096
DEFAULT_SCAN_SECONDS = 1.0
DEFAULT_MAX_CONTENT_CHARS = 32_768
DEFAULT_TASK_ROOTS = ("/tmp/selfEvolutionTask",)

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_TARGET_PATTERN = (
    r"(?:[：:]\s*)?"
    r"(?:[`\"']([^`\"'\r\n]{1,512})[`\"']|"
    r"([^\s,，。;；!?！？`\"']{1,500}?\.[A-Za-z0-9]{1,10}))"
)
_CHINESE_ENTRY = re.compile(
    r"^\s*(?:请\s*)?(?:阅读|读取|打开)\s*"
    r"(?:文件(?:\s+|[：:]\s*))?" + _TARGET_PATTERN
    + r"\s*(?:(?:并|然后|后)\s*(?:回答(?:问题)?|解答(?:问题)?|"
    r"完成(?:任务)?|处理(?:任务)?)|[，,]\s*获取(?:该)?任务信息)?"
    r"\s*[。！!]?\s*$"
)
_ENGLISH_ENTRY = re.compile(
    r"(?i)^\s*(?:read|open)(?:\s+|[：:]\s*)"
    r"(?:file(?:\s+|[：:]\s*))?" + _TARGET_PATTERN
    + r"\s*(?:and\s+(?:answer|solve)(?:\s+the\s+(?:question|task))?)?"
    r"\s*[.!]?\s*$"
)
_FILE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:/|\./|\.\./)?[^\s,，。;；!?！？`\"']{1,500}?\.[A-Za-z0-9]{1,10}"
)

_READER = r'''import os,stat,sys,time
mode,target=sys.argv[1],sys.argv[2]
max_depth,max_entries=int(sys.argv[3]),int(sys.argv[4])
scan_seconds,max_chars=float(sys.argv[5]),int(sys.argv[6])
task_roots=sys.argv[7:]
def fail(reason,detail=""):
 print("[TASK_INPUT_STATUS:error] "+reason+(": "+detail if detail else ""))
 raise SystemExit(2)
def safe_regular(path):
 try: info=os.stat(path,follow_symlinks=False)
 except FileNotFoundError: fail("not_found")
 except OSError as exc: fail("access_error",type(exc).__name__)
 if not stat.S_ISREG(info.st_mode): fail("not_regular")
 return path
if mode=="path":
 path=safe_regular(target)
else:
 roots=[os.path.abspath(os.getcwd())]
 def contains(parent,child):
  try: return os.path.commonpath((parent,child))==parent
  except ValueError: return False
 for candidate in task_roots:
  try: info=os.stat(candidate,follow_symlinks=False)
  except FileNotFoundError: continue
  except OSError as exc: fail("scan_incomplete",type(exc).__name__)
  if not stat.S_ISDIR(info.st_mode): continue
  candidate=os.path.abspath(candidate)
  if any(contains(root,candidate) for root in roots): continue
  roots=[root for root in roots if not contains(candidate,root)]
  roots.append(candidate)
 start=time.monotonic(); stack=[(root,0) for root in reversed(roots)]; matches=[]; rejected=0; seen=0
 def expired(): return time.monotonic()-start>scan_seconds
 while stack:
  if expired(): fail("scan_incomplete","time_limit")
  directory,depth=stack.pop()
  try:
   entries=[]
   with os.scandir(directory) as iterator:
    for entry in iterator:
     if expired(): fail("scan_incomplete","time_limit")
     seen+=1
     if seen>max_entries: fail("scan_incomplete","entry_limit")
     entries.append(entry)
  except OSError as exc: fail("scan_incomplete",type(exc).__name__)
  if expired(): fail("scan_incomplete","time_limit")
  entries.sort(key=lambda item:item.name)
  for entry in entries:
   if expired(): fail("scan_incomplete","time_limit")
   if entry.name==target:
    try: regular=entry.is_file(follow_symlinks=False)
    except OSError as exc: fail("scan_incomplete",type(exc).__name__)
    if regular and not any(ord(ch)<32 or ord(ch)==127 for ch in entry.path): matches.append(entry.path)
    else: rejected+=1
   try: is_dir=entry.is_dir(follow_symlinks=False)
   except OSError as exc: fail("scan_incomplete",type(exc).__name__)
   if is_dir:
    if depth>=max_depth: fail("scan_incomplete","depth_limit")
    stack.append((entry.path,depth+1))
 if expired(): fail("scan_incomplete","time_limit")
 if rejected: fail("not_regular" if not matches else "ambiguous")
 if not matches: fail("not_found")
 if len(matches)!=1: fail("ambiguous",str(len(matches)))
 path=safe_regular(matches[0])
try:
 with open(path,"r",encoding="utf-8",errors="strict") as handle: content=handle.read(max_chars+1)
except (OSError,UnicodeError) as exc: fail("read_error",type(exc).__name__)
print("[TASK_INPUT_PATH]")
print(os.path.abspath(path))
print("[TASK_INPUT_CONTENT]")
if len(content)>max_chars:
 print(content[:max_chars],end="")
 print("\n[TRUNCATED]")
 raise SystemExit(3)
print(content,end="")
'''


def extract_task_input(task_text: str) -> tuple[str, str] | None:
    """Return a conservative single read target, or None for normal LLM flow."""
    if (
        not isinstance(task_text, str)
        or not task_text
        or len(task_text) > 32_768
    ):
        return None
    normalized = task_text.strip()
    if not normalized or _CONTROL.search(normalized):
        return None
    match = _CHINESE_ENTRY.fullmatch(normalized)
    if match is None:
        match = _ENGLISH_ENTRY.fullmatch(normalized)
    references = list(_FILE_REFERENCE.finditer(normalized))
    if match is None or len(references) != 1:
        return None
    target = (match.group(1) or match.group(2)).strip()
    if (
        not target
        or len(target) > 512
        or _CONTROL.search(target)
        or target in (".", "..", "/")
    ):
        return None
    mode = "path" if "/" in target else "filename"
    if mode == "filename" and os.path.basename(target) != target:
        return None
    return mode, target


def build_task_input_command(
    mode: str,
    target: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    scan_seconds: float = DEFAULT_SCAN_SECONDS,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> str:
    if mode not in ("path", "filename"):
        raise ValueError("unsupported task input mode")
    if not target or len(target) > 512 or _CONTROL.search(target):
        raise ValueError("unsafe task input target")
    if not (0 <= max_depth <= 12 and 1 <= max_entries <= 10_000):
        raise ValueError("unsafe scan bounds")
    if not (0 < scan_seconds <= 5 and 1 <= max_content_chars <= 65_536):
        raise ValueError("unsafe read bounds")
    arguments = (
        mode,
        target,
        str(max_depth),
        str(max_entries),
        str(scan_seconds),
        str(max_content_chars),
        *DEFAULT_TASK_ROOTS,
    )
    command = "python3 -c " + shlex.quote(_READER) + " " + " ".join(
        shlex.quote(argument) for argument in arguments
    )
    if len(command) > MAX_TASK_INPUT_COMMAND_CHARS:
        raise ValueError("task input command exceeds platform limit")
    return command
