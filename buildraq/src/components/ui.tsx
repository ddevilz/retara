import type { ReactNode } from "react";
import { ArrowRight } from "lucide-react";

const EASE = "ease-[cubic-bezier(0.25,0.1,0.25,1)]";

/**
 * Text-roll hover animation: the label is duplicated inside an
 * overflow-hidden window; on `group-hover` the stack slides up by
 * exactly one line (-50% of the 2-line stack), swapping the visible copy.
 * Parent element must carry the `group` className.
 */
export function RollText({ text, className = "" }: { text: string; className?: string }) {
  return (
    <span className={`h-[20px] overflow-hidden ${className}`}>
      <span className={`flex flex-col transition-transform duration-500 ${EASE} group-hover:-translate-y-1/2`}>
        <span className="h-[20px] leading-[20px]">{text}</span>
        <span className="h-[20px] leading-[20px]">{text}</span>
      </span>
    </span>
  );
}

/**
 * Circle that rotates -45deg on hover, matching the arrow-button treatment
 * used across the hero nav CTA, hero/about CTAs, and mobile menu CTA.
 * Parent element must carry the `group` className.
 */
export function ArrowCircle({
  sizeClass,
  bgClass = "bg-white",
  iconClassName = "text-gray-900",
  iconSize = 14,
}: {
  sizeClass: string;
  bgClass?: string;
  iconClassName?: string;
  iconSize?: number;
}) {
  return (
    <span
      className={`flex items-center justify-center rounded-full transition-transform duration-500 ${EASE} group-hover:-rotate-45 ${sizeClass} ${bgClass}`}
    >
      <ArrowRight size={iconSize} className={iconClassName} />
    </span>
  );
}

/** Numbered-circle + pill-label pattern shared by the About and Case Studies section intros. */
export function BadgeRow({
  number,
  label,
  borderClass,
}: {
  number: string;
  label: string;
  borderClass: string;
}) {
  return (
    <div className="mb-6 flex items-center gap-3 px-5 sm:mb-8 sm:px-8 lg:px-12">
      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-900 text-[11px] font-semibold text-white sm:h-7 sm:w-7 sm:text-[12px]">
        {number}
      </span>
      <span
        className={`rounded-full border px-3 py-1 text-[12px] font-medium sm:px-4 sm:py-1.5 sm:text-[13px] ${borderClass}`}
      >
        {label}
      </span>
    </div>
  );
}

/**
 * Small circle that expands into a labeled pill on `group-hover` (the video
 * card hover buttons). The label is absolutely positioned so it never
 * affects the flex sizing of the collapsed circle; the icon anchors right.
 */
export function ExpandingHoverButton({
  variant,
  label,
  expandClass,
  icon,
}: {
  variant: "light" | "dark";
  label: string;
  /** literal `group-hover:w-[...]` class so Tailwind's content scanner can see it */
  expandClass: string;
  icon: ReactNode;
}) {
  const isDark = variant === "dark";
  return (
    <div
      className={`absolute bottom-4 left-4 flex h-9 w-9 items-center overflow-hidden rounded-full pr-2.5 transition-all duration-300 ease-in-out ${expandClass} ${
        isDark ? "bg-gray-900" : "bg-white"
      }`}
    >
      <span
        className={`absolute left-4 whitespace-nowrap text-[13px] font-medium opacity-0 transition-opacity duration-300 delay-100 group-hover:opacity-100 ${
          isDark ? "text-white" : "text-gray-900"
        }`}
      >
        {label}
      </span>
      <span className="ml-auto flex h-[14px] w-[14px] shrink-0 items-center justify-center">
        {icon}
      </span>
    </div>
  );
}
