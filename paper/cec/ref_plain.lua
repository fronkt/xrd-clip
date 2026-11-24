-- Run AFTER --citeproc. RSC style writes "ref. 18" in running text, not "ref.^18^":
-- a citation that directly follows "ref." / "refs." is turned from superscript into plain text.
local function is_ref_word(el)
  return el and el.t == "Str" and (el.text:match("ref%.$") or el.text:match("refs%.$"))
end

local function unsup(inls)
  local out = pandoc.List()
  for _, x in ipairs(inls) do
    if x.t == "Superscript" then out:extend(x.content) else out:insert(x) end
  end
  return out
end

function Inlines(inls)
  for i = 1, #inls do
    local el = inls[i]
    if el.t == "Cite" then
      local prev, prev2 = inls[i - 1], inls[i - 2]
      if is_ref_word(prev) then
        el.content = pandoc.List({ pandoc.Space() }) .. unsup(el.content)
      elseif prev and prev.t == "Space" and is_ref_word(prev2) then
        el.content = unsup(el.content)
      end
    end
  end
  return inls
end
