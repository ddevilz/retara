import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Clock, Menu, X } from "lucide-react";
import { RollText, ArrowCircle } from "./ui";

const NAV_LINKS = ["Products", "Studio", "Journal", "Connect"];

function formatLondonTime(): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/London",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
}

function useLondonTime(): string {
  const [time, setTime] = useState(formatLondonTime);
  useEffect(() => {
    const id = setInterval(() => setTime(formatLondonTime()), 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

/** Pill navbar shared by every page: logo → home, "Products" → /products (router Link), rest stay dead anchors. */
export default function Nav() {
  const londonTime = useLondonTime();
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  const closeMenu = () => setMenuOpen(false);

  return (
    <>
      <div className="relative z-20 mx-auto w-full max-w-[1440px] p-2 sm:p-3">
        <nav className="flex items-center justify-between rounded-full bg-white p-[5px]">
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className="flex h-9 w-9 items-center justify-center rounded-full bg-gray-900 text-[10px] font-bold tracking-tight text-white sm:h-10 sm:w-10 sm:text-[11px]"
            >
              BR
            </Link>
            <div className="hidden items-center gap-6 md:flex">
              {NAV_LINKS.map((link) =>
                link === "Products" ? (
                  <Link
                    key={link}
                    to="/products"
                    className="text-sm text-gray-900 transition-colors duration-300 hover:text-gray-500"
                  >
                    {link}
                  </Link>
                ) : (
                  <a
                    key={link}
                    href="#"
                    className="text-sm text-gray-900 transition-colors duration-300 hover:text-gray-500"
                  >
                    {link}
                  </a>
                ),
              )}
            </div>
          </div>

          <div className="hidden items-center gap-4 md:flex">
            <span className="hidden text-[13px] text-gray-600 lg:inline">
              Taking on projects for Q1 2026
            </span>
            <span className="flex items-center gap-1.5 text-[13px] text-gray-600">
              <Clock size={14} />
              {londonTime} in London
            </span>
            <button
              type="button"
              className="group flex items-center gap-3 rounded-full bg-gray-900 py-2 pl-5 pr-2 text-[13px] font-medium text-white"
            >
              <RollText text="Book a strategy call" />
              <ArrowCircle sizeClass="h-6 w-6" iconClassName="text-gray-900" iconSize={12} />
            </button>
          </div>

          <button
            type="button"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            onClick={() => setMenuOpen((v) => !v)}
            className="rounded-full bg-gray-900 p-3 text-white md:hidden"
          >
            {menuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </nav>
      </div>

      {/* Mobile menu overlay */}
      <div className={`fixed inset-0 z-50 md:hidden ${menuOpen ? "" : "pointer-events-none"}`}>
        <div
          className={`absolute inset-0 bg-black/60 transition-opacity duration-500 ${
            menuOpen ? "opacity-100" : "opacity-0"
          }`}
          onClick={closeMenu}
        />
        <div
          className={`absolute inset-x-0 bottom-0 mx-3 mb-3 rounded-2xl bg-white p-6 transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
            menuOpen ? "translate-y-0" : "translate-y-full"
          }`}
        >
          <div className="flex flex-col gap-8">
            <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-gray-200 px-3 py-1.5 text-[13px] text-gray-600">
              <Clock size={14} />
              {londonTime} in London
            </span>
            <nav className="flex flex-col gap-2">
              {NAV_LINKS.map((link) =>
                link === "Products" ? (
                  <Link
                    key={link}
                    to="/products"
                    onClick={closeMenu}
                    className="text-[28px] font-medium text-gray-900 sm:text-[32px]"
                  >
                    {link}
                  </Link>
                ) : (
                  <a
                    key={link}
                    href="#"
                    onClick={closeMenu}
                    className="text-[28px] font-medium text-gray-900 sm:text-[32px]"
                  >
                    {link}
                  </a>
                ),
              )}
            </nav>
            <button
              type="button"
              className="group flex w-fit items-center gap-3 rounded-full bg-[#F26522] py-2 pl-5 pr-2 text-[13px] font-medium text-white hover:bg-[#e05a1a] sm:pl-6 sm:text-[14px]"
            >
              <RollText text="Start a project" />
              <ArrowCircle sizeClass="h-7 w-7 sm:h-8 sm:w-8" iconClassName="text-[#F26522]" iconSize={16} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
