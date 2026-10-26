import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground shadow hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground shadow hover:bg-destructive/80",
        outline: "text-foreground border-white/20",
        cyan: "border-cyan-500/30 bg-cyan-500/15 text-cyan-400 font-bold tracking-wide uppercase",
        emerald: "border-emerald-500/30 bg-emerald-500/15 text-emerald-400 font-bold tracking-wide uppercase",
        crimson: "border-red-500/30 bg-red-500/15 text-red-400 font-bold tracking-wide uppercase",
        amber: "border-amber-500/30 bg-amber-500/15 text-amber-400 font-bold tracking-wide uppercase",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
