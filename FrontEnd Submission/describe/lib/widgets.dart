import 'package:flutter/material.dart';
import 'theme.dart';


//------------------------------------------------------------------------------
class AppLogo extends StatelessWidget {
  const AppLogo({super.key, this.size = 30});
  final double size;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: kYellow,
        borderRadius: BorderRadius.circular(size * 0.25),
      ),
      child: Icon(Icons.remove_red_eye_outlined,
          color: Colors.black, size: size * 0.6),
    );
  }
}


//------------------------------------------------------------------------------
class BigButton extends StatelessWidget {
  const BigButton({
    super.key,
    required this.color,
    required this.icon,
    required this.label,
    required this.onTap,
    this.hint,
    this.textColor = Colors.black,
  });

  final Color color;
  final Color textColor;
  final IconData icon;
  final String label;
  final String? hint;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: hint == null ? label : '$label. $hint',
      child: Material(
        color: color,
        borderRadius: BorderRadius.circular(10),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(10),
          child: Container(
            height: 56,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                Icon(icon, color: textColor, size: 22),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        label,
                        style: TextStyle(
                          color: textColor,
                          fontSize: 17,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      if (hint != null)
                        Text(
                          hint!,
                          style: TextStyle(
                            color: textColor.withValues(alpha: 0.7),
                            fontSize: 11,
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}


//------------------------------------------------------------------------------
class GhostButton extends StatelessWidget {
  const GhostButton({
    super.key,
    required this.label,
    required this.onTap,
    this.hint,
  });
  final String label;
  final VoidCallback onTap;

  /// Optional second line, used to advertise the matching voice command so the
  /// on-screen label and the spoken phrase cannot drift apart.
  final String? hint;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: hint == null ? 44 : 56,
      child: OutlinedButton(
        onPressed: onTap,
        style: OutlinedButton.styleFrom(
          backgroundColor: kCard,
          foregroundColor: Colors.white,
          side: BorderSide(color: borderColor()),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              label,
              style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
            ),
            if (hint != null)
              Text(
                hint!,
                style: const TextStyle(color: kMuted, fontSize: 11),
              ),
          ],
        ),
      ),
    );
  }
}


//------------------------------------------------------------------------------
class StatusBadge extends StatelessWidget {
  const StatusBadge(this.text, {super.key, required this.color});
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        border: Border.all(color: color),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}


//------------------------------------------------------------------------------
class TagChip extends StatelessWidget {
  const TagChip(this.text, {super.key});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0xFF2A2A2A),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(text,
          style: const TextStyle(color: Colors.white, fontSize: 10)),
    );
  }
}
